// Benchmark harness for the "prefix sum along x, then divide by (x+1)"
// pipeline from tutorial/lesson_25_inductive.cpp, compared against an
// optimized numpy implementation (bench_numpy.py) on the exact same data.
//
// Uses the fastest schedule from the lesson: prefix_sum is an inductive
// function fused into output's x loop via
// compute_at(x).store_at(y).fold_storage(x, 1) -- a single accumulator
// register per row, no materialized prefix_sum array at all. Single-core
// (no .parallel) to match a straight numpy comparison.
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

        Func input("input"), prefix_sum(Int(32), "prefix_sum"), output("output");
        input(x, y) = x + y;
        prefix_sum(x, y) = select(x <= 0, input(0, y), likely(prefix_sum(x - 1, y) + input(x, y)));
        output(x, y) = prefix_sum(x, y) / (x + 1);

        prefix_sum.compute_at(output, x).store_at(output, y).fold_storage(x, 1);
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

        printf("Halide (W=%d, H=%d): best of %d = %.3f ms (%.2f Mpixels/s)\n",
               W, H, trials, best_ms, (W * (double)H) / best_ms / 1000.0);

        // Dump the input array (same formula as `input(x, y) = x + y` above)
        // and the resulting output, so bench_numpy.py can run on identical
        // data and we can check for exact agreement.
        {
            std::vector<int32_t> in_flat(W * H);
            for (int yy = 0; yy < H; yy++)
                for (int xx = 0; xx < W; xx++)
                    in_flat[yy * W + xx] = xx + yy;

            std::vector<int32_t> out_flat(W * H);
            for (int yy = 0; yy < H; yy++)
                for (int xx = 0; xx < W; xx++)
                    out_flat[yy * W + xx] = result(xx, yy);

            FILE *f = fopen(data_path, "wb");
            int32_t header[2] = {W, H};
            fwrite(header, sizeof(int32_t), 2, f);
            fwrite(in_flat.data(), sizeof(int32_t), W * H, f);
            fwrite(out_flat.data(), sizeof(int32_t), W * H, f);
            fclose(f);
        }

        return 0;
    } catch (const Halide::Error &e) {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
}
