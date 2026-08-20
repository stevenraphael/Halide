// GPU prefix sum scheduled with NO parallelism in the scan (time) dimension --
// the schedule you are FORCED into when the scan's combine is non-associative.
//
// It is the exact same inclusive '+' prefix sum as prefixsum_gpu.cpp (same
// input, same output, validates against the same data), but where prefixsum_gpu
// parallelizes ALONG time via the two-stage chunk decomposition (up-sweep /
// down-sweep -- only valid because '+' is associative), here time is a single
// strictly serial recurrence per lane. A non-associative recurrence has no
// fixed-size associative monoid, so neither the two-stage nor CUB's decoupled
// look-back applies and this serial-time schedule is all you get. Parallelism
// is therefore ONLY across the S independent lanes.
//
// Memory layout is identical to the two-stage version: the lane index s is the
// contiguous (dim-0) axis, so the S threads of a block write coalesced. The
// point of the comparison is to isolate the value of scan-dim parallelism with
// everything else (layout, coalescing, fold) held equal.

#include "../support/bench_harness.h"
#include "Halide.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    int W = argc > 1 ? atoi(argv[1]) : 262144;  // per-lane sequence length (time)
    int S = argc > 2 ? atoi(argv[2]) : 1024;    // independent lanes (the ONLY parallel axis)
    const char *data_path = argc > 3 ? argv[3] : "/tmp/prefixsum_gpu_data.bin";

    try {
        Target target = get_host_target().with_feature(Target::CUDA).with_feature(Target::CUDACapability86);

        Var s("s"), t("t");
        // Real input in DRAM (lane s = dim0, contiguous) -- read from global memory
        // like CUB/Thrust do, so the memory traffic is a fair comparison (not an
        // inline-computed input that only writes).
        Buffer<int> in_buf(S, W);
        for (int tt = 0; tt < W; tt++)
            for (int ss = 0; ss < S; ss++)
                in_buf(ss, tt) = (tt + ss) & 255;

        Var so("so"), si("si");
        const int tf = std::min(128, S);

        // Two serial-time variants, both parallel over lanes only, both coalesced:
        //   inductive     : h_t = select(t<=0, x, likely(h_{t-1}+x_t)); h folds to
        //                   one accumulator per lane (O(S) state).
        //   non-inductive : h materialized via an explicit RDom scan over t
        //                   (O(W*S) trajectory), then copied out.
        auto build = [&](bool inductive) -> Func {
            Func h(Int(32), inductive ? "h_i" : "h_n"), out(inductive ? "out_i" : "out_n");
            if (inductive) {
                h(s, t) = select(t <= 0, in_buf(s, 0), likely(h(s, t - 1) + in_buf(s, t)));
                out(s, t) = h(s, t);
                out.bound(s, 0, S).bound(t, 0, W).reorder(t, s).gpu_tile(s, so, si, tf);
                h.compute_at(out, t).store_at(out, si).fold_storage(t, hb::fold_factor(1, W));
            } else {
                h(s, t) = in_buf(s, t);
                RDom rt(1, W - 1, "rt");
                h(s, rt) = h(s, rt - 1) + in_buf(s, rt);
                out(s, t) = h(s, t);
                // Materialize the full trajectory (O(W*S)) but in a SINGLE kernel:
                // the pure init is one 2-D GPU pass, and the time-scan update runs
                // the rt loop SERIALLY INSIDE the kernel (nested inside the lane
                // threads) -- reorder rt innermost, after the gpu split of s, so it
                // is not hoisted into a per-timestep relaunch.
                Var to("to"), ti("ti");
                h.compute_root().gpu_tile(s, t, so, to, si, ti, tf, 1);
                h.update(0).split(s, so, si, tf).gpu_blocks(so).gpu_threads(si).reorder(rt, si, so);
                out.bound(s, 0, S).bound(t, 0, W).reorder(s, t).gpu_tile(s, so, si, tf);
            }
            return out;
        };

        Func out_i = build(true), out_n = build(false);
        out_i.compile_jit(target);
        out_n.compile_jit(target);

        Buffer<int> res_i(S, W), res_n(S, W);  // dim0 = lane (contiguous)
        out_i.realize(res_i);
        out_n.realize(res_n);
        hb::Stats st_i = hb::bench([&] { out_i.realize(res_i); res_i.device_sync(); });
        hb::Stats st_n = hb::bench([&] { out_n.realize(res_n); res_n.device_sync(); });
        res_i.copy_to_host();
        res_n.copy_to_host();

        // Correctness vs a straight serial per-lane prefix sum.
        auto check = [&](Buffer<int> &r) {
            size_t m = 0;
            for (int ss = 0; ss < S; ss++) {
                int32_t run = 0;
                for (int tt = 0; tt < W; tt++) {
                    run += (tt + ss) & 255;
                    if (r(ss, tt) != run) m++;
                }
            }
            return m;
        };
        size_t mi = check(res_i), mn = check(res_n);
        Buffer<int> &result = res_i;
        size_t n_mismatch = mi;

        char note[192];
        snprintf(note, sizeof(note),
                 "GPU prefix sum, NO scan-dim parallelism (serial time / lane)  W=%d S=%d  "
                 "(the schedule a non-associative recurrence forces)",
                 W, S);
        hb::print_spec_header("prefixsum_gpu_serial", target.to_string(), note);
        // state_MB: inductive folds recurrence state to O(S); non-inductive
        // materializes the full O(W*S) trajectory.
        hb::print_row("non-inductive serial (materialize)", st_n,
                      (W * (double)S) / (st_n.min * 1e3), "Mpix/s",
                      (double)W * S * 4, (double)mn, mn == 0);
        hb::print_row("inductive serial (fold t->1)", st_i,
                      (W * (double)S) / (st_i.min * 1e3), "Mpix/s",
                      (double)S * 4, (double)mi, mi == 0,
                      hb::verdict(st_i.min, st_n.min));

        // Dump lane-major {input,output} so prefixsum_gpu_thrust.cu validates and
        // times the same data (identical to the two-stage version's output).
        std::vector<int32_t> in_flat((size_t)W * S), out_flat((size_t)W * S);
        for (int ss = 0; ss < S; ss++)
            for (int tt = 0; tt < W; tt++) {
                in_flat[(size_t)ss * W + tt] = (tt + ss) & 255;
                out_flat[(size_t)ss * W + tt] = result(ss, tt);
            }
        FILE *f = fopen(data_path, "wb");
        if (f) {
            int32_t header[2] = {W, S};
            fwrite(header, sizeof(int32_t), 2, f);
            fwrite(in_flat.data(), sizeof(int32_t), (size_t)W * S, f);
            fwrite(out_flat.data(), sizeof(int32_t), (size_t)W * S, f);
            fclose(f);
        }

        printf("  %s\n", n_mismatch == 0 ? "Success!" : "FAILED (scan mismatch)");
        return n_mismatch == 0 ? 0 : 1;
    } catch (const Halide::Error &e) {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
}
