// Benchmark an oneTBB tbb::parallel_scan implementation of the
// prefix-sum-then-average pipeline from tutorial/lesson_25_inductive.cpp,
// on the exact same data dumped by prefixsum_bench.cpp.
//
// Pipeline: input(x, y) = x + y
//           prefix_sum(x, y) = sum_{i<=x} input(i, y)
//           output(x, y) = prefix_sum(x, y) // (x + 1)
//
// Unlike bench_numpy.py's numpy/numba comparisons (deliberately pinned to
// one thread to match Halide's single-core schedule), this benchmark uses
// oneTBB's parallel_scan to fuse the running sum and the division into a
// single multi-threaded pass -- a genuinely optimized, parallel third-party
// baseline, rather than a single-core one.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <oneapi/tbb.h>

using namespace oneapi;

namespace {

// Uses oneTBB's lambda-based parallel_scan overload (backed internally by
// lambda_scan_body, so no hand-written Body class/reverse_join/assign/split
// constructor is needed here). The scan lambda computes a running int32 sum
// (matching Halide's Int(32) prefix_sum, including its silent overflow
// behavior for large W) and, on the final pass, fuses in the division by
// (x + 1) as a side effect, so no separate prefix_sum array is ever
// materialized; the return value only carries the running sum forward.
void fused_prefix_mean_tbb(const int32_t *inp, int32_t *out, int W, int H) {
    tbb::parallel_for(tbb::blocked_range<int>(0, H), [&](const tbb::blocked_range<int> &yr) {
        for (int y = yr.begin(); y < yr.end(); y++) {
            const int32_t *row_in = inp + (size_t)y * W;
            int32_t *row_out = out + (size_t)y * W;
            tbb::parallel_scan(
                tbb::blocked_range<int>(0, W), (int32_t)0,
                [&](const tbb::blocked_range<int> &r, int32_t sum, bool is_final_scan) -> int32_t {
                    int32_t temp = sum;
                    for (int i = r.begin(); i < r.end(); i++) {
                        temp = (int32_t)(temp + row_in[i]);
                        if (is_final_scan) {
                            // temp can silently overflow into negative
                            // int32 values for large x (matching Halide's
                            // Int(32) prefix_sum wraparound). Halide's `/`
                            // on signed integers is floor division, while
                            // C++'s `/` truncates toward zero, so they
                            // disagree whenever temp is negative and not
                            // an exact multiple of the divisor -- correct
                            // for that explicitly.
                            int32_t divisor = i + 1;
                            int32_t q = temp / divisor;
                            if (temp < 0 && temp % divisor != 0) {
                                q -= 1;
                            }
                            row_out[i] = q;
                        }
                    }
                    return temp;
                },
                [](int32_t a, int32_t b) -> int32_t { return (int32_t)(a + b); });
        }
    });
}

}  // namespace

int main(int argc, char **argv) {
    const char *data_path = argc > 1 ? argv[1] : "/tmp/prefixsum_bench_data.bin";

    FILE *f = fopen(data_path, "rb");
    if (!f) {
        fprintf(stderr, "Could not open %s (run prefixsum_bench first)\n", data_path);
        return 1;
    }
    int32_t header[2];
    if (fread(header, sizeof(int32_t), 2, f) != 2) {
        fprintf(stderr, "Bad header in %s\n", data_path);
        return 1;
    }
    int W = header[0], H = header[1];

    std::vector<int32_t> inp((size_t)W * H), halide_out((size_t)W * H), out((size_t)W * H);
    if (fread(inp.data(), sizeof(int32_t), (size_t)W * H, f) != (size_t)W * H ||
        fread(halide_out.data(), sizeof(int32_t), (size_t)W * H, f) != (size_t)W * H) {
        fprintf(stderr, "Bad data in %s\n", data_path);
        return 1;
    }
    fclose(f);

    fused_prefix_mean_tbb(inp.data(), out.data(), W, H);  // warm-up.

    const int trials = 10;
    double best_ms = 1e18;
    for (int i = 0; i < trials; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        fused_prefix_mean_tbb(inp.data(), out.data(), W, H);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best_ms) best_ms = ms;
    }

    printf("tbb       (W=%d, H=%d): best of %d = %.3f ms (%.2f Mpixels/s)\n",
           W, H, trials, best_ms, (W * (double)H) / best_ms / 1000.0);

    size_t n_mismatch = 0;
    for (size_t i = 0; i < out.size(); i++) {
        if (out[i] != halide_out[i]) n_mismatch++;
    }
    printf("Output mismatch vs Halide: %zu / %zu (%s)\n",
           n_mismatch, out.size(), n_mismatch == 0 ? "OK" : "DIFFERS");

    return 0;
}
