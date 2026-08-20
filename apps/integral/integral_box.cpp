// Integral image (summed-area table) -> box filter, inductive vs non-inductive.
//
// SAT = 2D prefix sum (clean, separable, single-component scans: rowsum along x,
// sat along y). The box filter consumes the SAT in-order within a bounded +/-R
// window (reads a little ahead, no reverse pass), so:
//   inductive : rowsum/sat are inductive funcs fused into the box filter and
//               folded -- sat keeps a band of ~2R+2 rows, never the full image.
//   noninduct : rowsum/sat are RDom prefix sums; an RDom scan can't fuse into
//               the box filter, so the FULL W*H integral image is materialized
//               (this is what OpenCV integral() produces).
// Both compute the identical box output. Single-thread (the SAT y-scan is serial
// either way) so the comparison isolates the memory/fold effect.
//
// Validated against OpenCV boxFilter (see integral_cv.py).
//
// Build: g++ apps/integral/integral_box.cpp -O3 -march=native -Iinclude
//        -Lbuild/src -lHalide -lpthread -ldl -o /tmp/ib -std=c++17
//        LD_LIBRARY_PATH=build/src /tmp/ib [W H R]

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    int W = argc > 1 ? atoi(argv[1]) : 4096;
    int H = argc > 2 ? atoi(argv[2]) : 4096;
    int R = argc > 3 ? atoi(argv[3]) : 8;   // window radius (box is (2R+1)^2)

    Buffer<float> in(W, H);
    srand(9);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) in(x, y) = (float)(rand() % 256);

    auto build_box = [&](bool inductive) -> Func {
        Var x("x"), y("y");
        Func input("input");
        input(x, y) = cast<double>(in(clamp(x, 0, W - 1), clamp(y, 0, H - 1)));

        Func rowsum(Float(64), "rowsum"), sat(Float(64), "sat");
        RDom rx(1, W - 1), ry(1, H - 1);   // hoisted so scheduling can see them
        if (inductive) {
            rowsum(x, y) = select(x <= 0, input(0, y), likely(rowsum(x - 1, y) + input(x, y)));
            sat(x, y) = select(y <= 0, rowsum(x, 0), likely(sat(x, y - 1) + rowsum(x, y)));
        } else {
            rowsum(x, y) = undef<double>();
            rowsum(0, y) = input(0, y);
            rowsum(rx, y) = rowsum(rx - 1, y) + input(rx, y);
            sat(x, y) = undef<double>();
            sat(x, 0) = rowsum(x, 0);
            sat(x, ry) = sat(x, ry - 1) + rowsum(x, ry);
        }

        Func satc("satc");
        satc(x, y) = select(x < 0 || y < 0, Expr(0.0), sat(clamp(x, 0, W - 1), clamp(y, 0, H - 1)));
        Func box("box");
        box(x, y) = satc(x + R, y + R) - satc(x - R - 1, y + R)
                  - satc(x + R, y - R - 1) + satc(x - R - 1, y - R - 1);
        box.bound(x, 0, W).bound(y, 0, H);

        if (inductive) {
            // Fuse SAT into the box filter, fold to a band -- NEVER materialized.
            sat.store_root().compute_at(box, y).fold_storage(y, 2 * R + 2);
            rowsum.store_root().compute_at(box, y).fold_storage(x, 1);
            box.vectorize(x, 8);   // serial y (band slides), vectorize x
        } else {
            // Full W*H integral image materialized, but WELL scheduled:
            // horizontal pass parallel over rows; vertical pass with x inner
            // (cache-friendly sequential access) + vectorized + parallel tiles.
            rowsum.compute_root();
            rowsum.update(1).parallel(y);
            sat.compute_root();
            Var xo("xo"), xi("xi");
            sat.update(1).split(x, xo, xi, 64).reorder(xi, ry, xo).vectorize(xi, 8).parallel(xo);
            box.parallel(y).vectorize(x, 8);
        }
        return box;
    };

    try {
        Func bi = build_box(true), bn = build_box(false);
        bi.compile_jit();
        bn.compile_jit();
        Buffer<double> ri(W, H), rn(W, H);
        bi.realize(ri);
        bn.realize(rn);

        auto bench = [&](Func f, Buffer<double> &b) {
            double best = 1e18;
            for (int i = 0; i < 5; i++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                f.realize(b);
                auto t1 = std::chrono::high_resolution_clock::now();
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            return best;
        };
        double t_i = bench(bi, ri), t_n = bench(bn, rn);

        double err = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) err = std::max(err, (double)std::abs(ri(x, y) - rn(x, y)));

        double sat_full = (double)W * H * 8 / (1024.0 * 1024.0);
        double sat_band = (double)W * (2 * R + 2) * 8 / 1024.0;
        printf("Integral image -> box filter  W=%d H=%d R=%d (window %dx%d)\n",
               W, H, R, 2 * R + 1, 2 * R + 1);
        printf("  inductive (SAT folded to band, %.0f KB): %8.3f ms\n", sat_band, t_i);
        printf("  non-inductive (SAT materialized, %.0f MB): %8.3f ms\n", sat_full, t_n);
        printf("  |inductive - non-inductive| = %.3g\n", err);

        // Dump input + inductive box output (binary) for OpenCV validation.
        FILE *fi = fopen("apps/integral/in.bin", "wb");
        FILE *fb = fopen("apps/integral/box.bin", "wb");
        FILE *fp = fopen("apps/integral/params.txt", "w");
        if (fi && fb && fp) {
            fwrite(in.data(), sizeof(float), (size_t)W * H, fi);
            std::vector<double> tmp((size_t)W * H);
            for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) tmp[(size_t)y * W + x] = ri(x, y);
            fwrite(tmp.data(), sizeof(double), (size_t)W * H, fb);
            fprintf(fp, "%d %d %d\n", W, H, R);
        }
        if (fi) fclose(fi); if (fb) fclose(fb); if (fp) fclose(fp);
        return err < 1e-1 ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
