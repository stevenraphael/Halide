// One-stage prefix-sum: simple inductive fold over time, parallel over the
// INDEPENDENT LANES only (S-way parallelism). Serial scan per lane. This is the
// variant that loses to oneTBB when the lane count is small, because it has no
// time-parallelism -- motivating the two-stage (time-parallel) scan.
#include "../support/bench_harness.h"
#include "Halide.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int W = argc > 1 ? atoi(argv[1]) : 1048576;  // time length
    int S = argc > 2 ? atoi(argv[2]) : 32;       // independent lanes (parallel axis)
    const char *data_path = argc > 3 ? argv[3] : "/tmp/prefixsum_bench_data.bin";

    try {
        Var t("t"), s("s");
        Func input("input"), prefix_sum(Int(32), "prefix_sum"), output("output");
        input(t, s) = (t + s) & 255;  // bounded: no int32 overflow
        prefix_sum(t, s) = select(t <= 0, input(0, s), likely(prefix_sum(t - 1, s) + input(t, s)));
        output(t, s) = prefix_sum(t, s) / (t + 1);

        prefix_sum.compute_at(output, t).store_at(output, s).fold_storage(t, hb::fold_factor(1, W));
        output.bound(t, 0, W).bound(s, 0, S).reorder(t, s).parallel(s);  // parallel over lanes

        Buffer<int> result(W, S);
        output.realize(result);
        hb::Stats sb = hb::bench([&] { output.realize(result); });

        // UNFOLD=1 pins fold_storage(t) to W: same fusion, materializes the full
        // O(W*S) prefix trajectory -> report the unfolded label + footprint.
        const bool unfolded = getenv("UNFOLD") != nullptr;
        const double fp_unfold = (double)W * S * 4;
        char note[160];
        snprintf(note, sizeof(note),
                 "One-stage fold, parallel over lanes  W=%d S=%d  |  unfolded fp/LLC=%.3f",
                 W, S, hb::footprint_over_llc(fp_unfold));
        hb::print_spec_header("prefixsum_one_stage", "host", note);
        hb::print_row(unfolded ? "one-stage UNFOLDED (fold t -> W)" : "one-stage FOLDED (parallel lanes)", sb,
                      (W * (double)S) / (sb.min * 1e3), "Mpix/s",
                      unfolded ? fp_unfold : (double)S * 4, 0.0, true, "", fp_unfold);

        // Dump same format for the oneTBB partner.
        std::vector<int32_t> in_flat((size_t)W * S), out_flat((size_t)W * S);
        for (int ss = 0; ss < S; ss++)
            for (int tt = 0; tt < W; tt++) {
                in_flat[(size_t)ss * W + tt] = (tt + ss) & 255;
                out_flat[(size_t)ss * W + tt] = result(tt, ss);
            }
        FILE *f = fopen(data_path, "wb");
        int32_t header[2] = {W, S};
        fwrite(header, sizeof(int32_t), 2, f);
        fwrite(in_flat.data(), sizeof(int32_t), (size_t)W * S, f);
        fwrite(out_flat.data(), sizeof(int32_t), (size_t)W * S, f);
        fclose(f);
        return 0;
    } catch (const Halide::Error &e) {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
}
