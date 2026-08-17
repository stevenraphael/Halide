// GPU batched prefix scan (inclusive sum) expressed with inductive functions,
// benchmarked against a vendor library (CUB/Thrust) via prefixsum_gpu_thrust.cu
// and against the serial-scan-dim schedule in prefixsum_gpu_nonassoc.cpp.
//
// Workload: S independent lanes, each an inclusive prefix sum over a length-W
// sequence (a segmented / batched scan). Because '+' is associative we can
// parallelize ALONG time with the two-stage chunk decomposition, t = k*L + j:
//   input(s,j,k) = ((k*L+j)+s) & 255            bounded so int32 never overflows
//   ctot(s,k)    = sum_{r<L} input(s,r,k)         up-sweep: per-chunk total (parallel)
//   carry(s,k)   = sum_{q<k} ctot(s,q)            serial O(C) exclusive chunk prefix
//   local(s,j,k) = prefix_{i<=j} input(s,i,k)     inductive over j, folded to O(1)
//   out(s,j,k)   = carry(s,k) + local(s,j,k)      down-sweep: add inter-chunk carry
//
// GPU map: chunk k -> gpu_blocks, lane s -> gpu_threads; the j-scan runs serially
// inside each thread with local folded to a single accumulator. carry keeps its
// short serial O(C) scan (like CUB's inter-block step), parallel over lanes.
//
// Lane s is the contiguous (dim-0) storage axis so the block's threads write
// coalesced -- identical layout to prefixsum_gpu_nonassoc.cpp, so the only thing
// that differs between the two is whether the scan dimension is parallelized.

#include "../support/bench_harness.h"
#include "Halide.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    int W = argc > 1 ? atoi(argv[1]) : 262144;  // per-lane sequence length (time)
    int S = argc > 2 ? atoi(argv[2]) : 1024;    // independent lanes (batch/segments)
    int L = argc > 3 ? atoi(argv[3]) : 1024;    // chunk length (serial fold depth)
    const char *data_path = argc > 4 ? argv[4] : "/tmp/prefixsum_gpu_data.bin";

    if (W % L != 0) {
        W = (W / L) * L;
        if (W == 0) {
            fprintf(stderr, "W must be >= L\n");
            return 1;
        }
    }
    int C = W / L;  // number of time chunks (parallel across GPU blocks)

    try {
        Target target = get_host_target().with_feature(Target::CUDA).with_feature(Target::CUDACapability89);

        Var s("s"), j("j"), k("k");
        // Real input in DRAM (lane s = dim0, contiguous) so memory traffic matches
        // CUB/Thrust (which read d_in): the two-stage reads it twice (up + down).
        Buffer<int> in_buf(S, W);
        for (int tt = 0; tt < W; tt++)
            for (int ss = 0; ss < S; ss++)
                in_buf(ss, tt) = (tt + ss) & 255;

        Func input("input"), ctot("ctot"), carry(Int(32), "carry"),
            local(Int(32), "local"), output("output");
        input(s, j, k) = in_buf(s, k * L + j);

        // Up-sweep: per-chunk total (explicit reduction; parallel over k and s).
        RDom r(0, L, "r");
        ctot(s, k) = cast<int>(0);
        ctot(s, k) += input(s, r, k);

        // Serial O(C) exclusive prefix over chunk totals, parallel over lanes.
        carry(s, k) = select(k <= 0, 0, likely(carry(s, k - 1) + ctot(s, k - 1)));

        // Down-sweep: per-chunk inductive local prefix (folded), plus the carry.
        local(s, j, k) = select(j <= 0, input(s, 0, k),
                                likely(local(s, j - 1, k) + input(s, j, k)));
        output(s, j, k) = carry(s, k) + local(s, j, k);

        // ---- GPU schedule (lane s = contiguous dim0 => coalesced writes) ----
        Var so("so"), si("si");
        // Up-sweep reduction: chunk k -> blocks, lane s -> threads; r serial.
        ctot.compute_root().gpu_blocks(k).gpu_threads(s);
        ctot.update(0).reorder(r, s, k).gpu_blocks(k).gpu_threads(s);
        // Serial chunk-prefix: parallel over lanes, k serial inside each thread.
        carry.compute_root().reorder(k, s).gpu_tile(s, so, si, std::min(128, S));
        // Down-sweep: chunk k -> blocks, lane s -> threads; j serial (the fold).
        output.bound(s, 0, S).bound(j, 0, L).bound(k, 0, C).reorder(j, s, k).gpu_blocks(k).gpu_threads(s);
        local.compute_at(output, j).store_at(output, s).fold_storage(j, hb::fold_factor(1, L));

        output.compile_jit(target);

        Buffer<int> result(S, L, C);  // dim0 = lane (contiguous)
        output.realize(result);
        result.copy_to_host();

        hb::Stats st = hb::bench([&] {
            output.realize(result);
            result.device_sync();
        });
        result.copy_to_host();

        // Correctness vs a straight serial per-lane prefix sum on the host.
        size_t n_mismatch = 0;
        for (int ss = 0; ss < S; ss++) {
            int32_t run = 0;
            for (int tt = 0; tt < W; tt++) {
                run += (tt + ss) & 255;
                if (result(ss, tt % L, tt / L) != run) n_mismatch++;
            }
        }

        const double bytes_fold = (double)C * S * 3 * 4;  // ctot+carry+fold accum
        char note[192];
        snprintf(note, sizeof(note),
                 "GPU batched inclusive scan  W=%d S=%d L=%d C=%d  (inductive 2-stage, "
                 "scan-dim parallel: blocks=chunks threads=lanes, fold j->1)",
                 W, S, L, C);
        hb::print_spec_header("prefixsum_gpu", target.to_string(), note);
        hb::print_row("Halide 2-stage (scan-dim parallel, GPU)", st,
                      (W * (double)S) / (st.min * 1e3), "Mpix/s",
                      bytes_fold, (double)n_mismatch, n_mismatch == 0);

        // Dump lane-major {input,output} for the vendor (thrust/CUB) comparison.
        std::vector<int32_t> in_flat((size_t)W * S), out_flat((size_t)W * S);
        for (int ss = 0; ss < S; ss++)
            for (int tt = 0; tt < W; tt++) {
                in_flat[(size_t)ss * W + tt] = (tt + ss) & 255;
                out_flat[(size_t)ss * W + tt] = result(ss, tt % L, tt / L);
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
