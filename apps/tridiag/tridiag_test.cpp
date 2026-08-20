// Scalar tridiagonal solve (Thomas algorithm) across H independent lines of
// length W -- one directional sweep of an ADI/NPB-BT-style solver -- built
// inductive vs non-inductive, then fed into an in-order same-axis consumer.
//
// Per line: forward elimination produces cp,dp (scan increasing i); back
// substitution produces the solution (scan decreasing i). Then a consumer does
// a prefix sum of the solution along i (stands in for the next in-order stage).
//
//   inductive : cp,dp,xr are inductive funcs; consumer ps is inductive too.
//   noninduct : cp,dp,xr and ps are RDom update-def scans.
// Both are scheduled per line (compute_at the line loop), so each keeps only
// one line live. The question the benchmark answers: does the reverse
// back-substitution (which reads a whole line of cp,dp) leave any room for the
// inductive O(1) fold, or does it force full-line storage for both?
//
// Build: g++ apps/tridiag/tridiag_test.cpp -Iinclude -Lbuild/src -lHalide
//        -lpthread -ldl -o /tmp/tri -std=c++17 ; LD_LIBRARY_PATH=build/src /tmp/tri

#include "Halide.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    int W = argc > 1 ? atoi(argv[1]) : 2048;   // line length (solve axis)
    int H = argc > 2 ? atoi(argv[2]) : 4096;   // number of independent lines

    // Diagonally dominant tridiagonal system per line: a=c=-1, b=4, d random.
    srand(7);
    Buffer<float> a(W, H), b(W, H), c(W, H), d(W, H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            a(x, y) = (x == 0) ? 0.0f : -1.0f;
            c(x, y) = (x == W - 1) ? 0.0f : -1.0f;
            b(x, y) = 4.0f;
            d(x, y) = (float)(rand() % 100) / 50.0f - 1.0f;
        }

    try {
        Var x("x"), y("y");

        auto build = [&](bool inductive) -> Func {
            Func cp(Float(32), "cp"), dp(Float(32), "dp");
            Func xr(Float(32), "xr");            // back-sub in reversed index m
            Func sol("sol"), ps(Float(32), "ps"), out("out");

            if (inductive) {
                cp(x, y) = select(x <= 0, c(0, y) / b(0, y),
                                  likely(c(x, y) / (b(x, y) - a(x, y) * cp(x - 1, y))));
                dp(x, y) = select(x <= 0, d(0, y) / b(0, y),
                                  likely((d(x, y) - a(x, y) * dp(x - 1, y)) /
                                         (b(x, y) - a(x, y) * cp(x - 1, y))));
                // m = reversed index; actual index W-1-m. xr(0)=dp(W-1).
                Var m("m");
                xr(m, y) = select(m <= 0, dp(W - 1, y),
                                  likely(dp(W - 1 - m, y) -
                                         cp(W - 1 - m, y) * xr(m - 1, y)));
                sol(x, y) = xr(W - 1 - x, y);
                ps(x, y) = select(x <= 0, sol(0, y), likely(ps(x - 1, y) + sol(x, y)));
                out(x, y) = ps(x, y);
                // Per-line schedule; back-sub reads a whole line of cp/dp/xr.
                cp.compute_at(out, y).store_at(out, y);
                dp.compute_at(out, y).store_at(out, y);
                xr.compute_at(out, y).store_at(out, y);
                sol.compute_at(out, y);
                ps.compute_at(out, y).store_at(out, y).fold_storage(x, 2);
            } else {
                RDom rx(1, W - 1);
                cp(x, y) = undef<float>();
                cp(0, y) = c(0, y) / b(0, y);
                cp(rx, y) = c(rx, y) / (b(rx, y) - a(rx, y) * cp(rx - 1, y));
                dp(x, y) = undef<float>();
                dp(0, y) = d(0, y) / b(0, y);
                dp(rx, y) = (d(rx, y) - a(rx, y) * dp(rx - 1, y)) /
                            (b(rx, y) - a(rx, y) * cp(rx - 1, y));
                sol(x, y) = undef<float>();
                sol(W - 1, y) = dp(W - 1, y);
                RDom rb(0, W - 1);               // back-sub: i = W-2 .. 0
                Expr ib = W - 2 - rb;
                sol(ib, y) = dp(ib, y) - cp(ib, y) * sol(ib + 1, y);
                ps(x, y) = undef<float>();
                ps(0, y) = sol(0, y);
                RDom rp(1, W - 1);
                ps(rp, y) = ps(rp - 1, y) + sol(rp, y);
                out(x, y) = ps(x, y);
                cp.compute_at(out, y);
                dp.compute_at(out, y);
                sol.compute_at(out, y);
                ps.compute_at(out, y);
            }
            // Lines are independent -> parallelize across y. The recurrence
            // stays serial within a line (along x).
            out.bound(x, 0, W).bound(y, 0, H).parallel(y);
            return out;
        };

        Func out_ind = build(true), out_rd = build(false);
        out_ind.compile_jit();
        out_rd.compile_jit();
        Buffer<float> bi(W, H), br(W, H);
        out_ind.realize(bi);
        out_rd.realize(br);

        auto bench = [&](Func f, Buffer<float> &b_) {
            double best = 1e18;
            for (int t = 0; t < 10; t++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                f.realize(b_);
                auto t1 = std::chrono::high_resolution_clock::now();
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            return best;
        };
        double t_ind = bench(out_ind, bi), t_rd = bench(out_rd, br);

        // C++ Thomas ground truth + prefix sum.
        double err = 0;
        std::vector<float> cpg(W), dpg(W), xg(W);
        for (int y = 0; y < H; y++) {
            cpg[0] = c(0, y) / b(0, y);
            dpg[0] = d(0, y) / b(0, y);
            for (int i = 1; i < W; i++) {
                float den = b(i, y) - a(i, y) * cpg[i - 1];
                cpg[i] = c(i, y) / den;
                dpg[i] = (d(i, y) - a(i, y) * dpg[i - 1]) / den;
            }
            xg[W - 1] = dpg[W - 1];
            for (int i = W - 2; i >= 0; i--) xg[i] = dpg[i] - cpg[i] * xg[i + 1];
            float run = 0;
            for (int i = 0; i < W; i++) {
                run += xg[i];
                err = std::max({err, (double)std::abs(bi(i, y) - run),
                                     (double)std::abs(br(i, y) - run)});
            }
        }

        printf("Tridiagonal solve + prefix-sum consumer  W=%d H=%d\n", W, H);
        printf("  inductive:     %7.3f ms\n", t_ind);
        printf("  non-inductive: %7.3f ms\n", t_rd);
        printf("  max abs err %.3g -> %s\n", err, err < 1e-2 ? "PASS" : "FAIL");
        return err < 1e-2 ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
