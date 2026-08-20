// Benchmark harness for the "prefix sum along x, then divide by (x+1)"
// pipeline expressed with a plain RDom scan (the non-inductive idiom from
// tutorial/lesson_25_inductive.cpp: prefix_sum(x, y) = undef<int>(), with
// prefix_sum(0, y) and prefix_sum(r, y) update definitions), instead of the
// inductive-function version in prefixsum_bench.cpp.
//
// Unlike the inductive schedule, an RDom forces the entire scan over r to
// complete before output can be computed, so prefix_sum.compute_at(output, y)
// materializes a full row of prefix_sum (not folded to a single register).
// Single-core (no .parallel), same as prefixsum_bench.cpp, so the two are
// directly comparable.
#include "Halide.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    int W = argc > 1 ? atoi(argv[1]) : 65536;
    int H = argc > 2 ? atoi(argv[2]) : 32;
    const char *data_path = argc > 3 ? argv[3] : "/tmp/prefixsum_bench_data.bin";

    try {
        Var x("x"), y("y");

        Func input("input"), prefix_sum("prefix_sum"), output("output");
        input(x, y) = x + y;

        RDom r(1, W - 1, "r");
        prefix_sum(x, y) = undef<int>();
        prefix_sum(0, y) = input(0, y);
        prefix_sum(r, y) = prefix_sum(r - 1, y) + input(r, y);

        output(x, y) = prefix_sum(x, y) / (x + 1);

        prefix_sum.compute_at(output, y);
        output.bound(x, 0, W).bound(y, 0, H);

        Buffer<int> result(W, H);
        output.realize(result);  // warm-up / JIT compile.

        const int trials = 10;
        double best_ms = 1e18;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            output.realize(result);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best_ms) best_ms = ms;
        }

        printf("Halide-RDom (W=%d, H=%d): best of %d = %.3f ms (%.2f Mpixels/s)\n",
               W, H, trials, best_ms, (W * (double)H) / best_ms / 1000.0);

        // Compare against the reference data dumped by prefixsum_bench.cpp.
        FILE *f = fopen(data_path, "rb");
        if (f) {
            int32_t header[2];
            fread(header, sizeof(int32_t), 2, f);
            std::vector<int32_t> in_flat(W * H), halide_out(W * H);
            fread(in_flat.data(), sizeof(int32_t), W * H, f);
            fread(halide_out.data(), sizeof(int32_t), W * H, f);
            fclose(f);

            int n_mismatch = 0;
            for (int yy = 0; yy < H; yy++)
                for (int xx = 0; xx < W; xx++)
                    if (result(xx, yy) != halide_out[yy * W + xx]) n_mismatch++;
            printf("Output mismatch vs Halide (inductive): %d / %d (%s)\n",
                   n_mismatch, W * H, n_mismatch == 0 ? "OK" : "DIFFERS");
        }

        return 0;
    } catch (const Halide::Error &e) {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
}
