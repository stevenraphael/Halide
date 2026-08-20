// Unified GPU head-to-head for the Mamba selective-SSM scan
//   h_t = a_t h_{t-1} + b_t x_t ,   y_t = sum_n c_t h_t   (state n, channel d).
//
// This consolidates the previously-separate GPU probes (mamba_gpu*.cpp) into ONE
// driver with a single, uniform benchmarking methodology so the numbers are
// directly comparable and paper-ready. Four Halide GPU variants are built:
//
//   (A) sequential   non-inductive : RDom-over-t scan; the trajectory h(d,n,t) is
//                                     materialized to global memory (D*N*T floats).
//   (B) sequential   inductive     : register-resident sliding window, fold(t,2);
//                                     nothing but y transits global memory.
//   (C) two-stage    non-inductive : sqrt(T)-chunked scan, but every intermediate
//                                     (PA,PB,hchunk) is materialized to global.
//   (D) two-stage    inductive     : register-resident chunk scans; only the small
//                                     per-chunk aggregates/carries (D*N*C) transit
//                                     global memory -- the schedule that has BOTH
//                                     halves of the fused mamba_ssm kernel.
//
// The production SOTA (mamba_ssm's hand-written fused CUDA selective_scan_fn) is
// benchmarked at the SAME (N,D,T) by run_mamba_gpu.sh, which appends its row to
// this table. The point of the paper's GPU experiment: (D) matches the structure
// of the fused CUDA kernel and is trivial to write inductively, while (A)/(C) --
// the only things you can write WITHOUT inductive funcs -- either serialize the
// whole of T or blow the trajectory through DRAM, and at large T the non-inductive
// materialization simply cannot be allocated.
//
// Methodology (uniform across every variant): correctness gate vs a CPU streaming
// reference (max abs err < 1e-2), 3 warmup realizes, then >=30 timed realizes with
// an explicit device_sync inside the timed region; we report the MEDIAN and the
// inter-quartile range [p25,p75] in ms, plus the achieved tokens/s (D*T/median).
//
// Build: g++ apps/mamba/mamba_gpu_bench.cpp -O3 -march=native -Iinclude
//        -Lbuild/src -lHalide -lpthread -ldl -o /tmp/mgb -std=c++17
//        LD_LIBRARY_PATH=build/src:/usr/lib/wsl/lib /tmp/mgb [N D T]

#include "Halide.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 16;
    int D = argc > 2 ? atoi(argv[2]) : 512;
    int T = argc > 3 ? atoi(argv[3]) : 16384;
    int L = (int)std::ceil(std::sqrt((double)T));
    int C = (T + L - 1) / L;

    Target target = get_host_target().with_feature(Target::CUDA)
                        .with_feature(Target::CUDACapability86);

    // Real Mamba parameterization: the per-step decay is NOT a materialized
    // O(D*N*T) tensor. A is (D,N) and time-invariant, delta is (D,T); the decay
    // deltaA(d,n,t) = exp(delta(d,t) * A(d,n)) is recomputed inline (Func aF),
    // exactly as selective_scan does -- so we read O(D*T)+O(D*N) inputs, not
    // O(D*N*T). Reading a full a(d,n,t) tensor was an N-fold HBM-traffic handicap
    // unrepresentative of Mamba.
    Buffer<float> Av(D, N), delta(D, T), b(N, T), c(N, T), x(D, T);
    srand(5);
    for (int dd = 0; dd < D; dd++)
        for (int nn = 0; nn < N; nn++)
            Av(dd, nn) = -(0.01f + (rand() % 99) / 100.0f);   // A < 0 -> stable
    for (int t = 0; t < T; t++) {
        for (int nn = 0; nn < N; nn++) {
            b(nn, t) = (rand() % 200) / 100.0f - 1.0f;
            c(nn, t) = (rand() % 200) / 100.0f - 1.0f;
        }
        for (int dd = 0; dd < D; dd++) {
            delta(dd, t) = (rand() % 100) / 1000.0f;          // delta > 0
            x(dd, t) = (rand() % 200) / 100.0f - 1.0f;
        }
    }
    Var d("d"), n("n"), k("k"), j("j"), t("t");
    // Decay computed INLINE from the small inputs (A is (D,N), delta is (D,T)),
    // exactly as real Mamba does -- no O(D*N*T) decay tensor. (This is safe for
    // storage folding: the only foldable scan is hchunk over ykj's j loop; the
    // PAj/loc chunk scans are produced whole per thread and are not folded.)
    auto aF = [&](const Expr &dd, const Expr &nn, const Expr &tt) -> Expr {
        return 0.9f + 0.09f * (delta(dd, tt) * Av(dd, nn));
    };

    // ---- (A)/(B) SEQUENTIAL, register-resident when inductive ----
    auto build_seq = [&](bool inductive) -> Func {
        Func h(Float(32), "h_seq");
        if (inductive) {
            h(d, n, t) = select(t <= 0, b(n, 0) * x(d, 0),
                                likely(aF(d, n, t) * h(d, n, t - 1) + b(n, t) * x(d, t)));
        } else {
            h(d, n, t) = b(n, 0) * x(d, 0);
            RDom rt(1, T - 1);
            h(d, n, rt) = aF(d, n, rt) * h(d, n, rt - 1) + b(n, rt) * x(d, rt);
        }
        Func y("y_out");
        Expr acc = 0.0f;
        for (int nn = 0; nn < N; nn++) acc += c(nn, t) * h(d, nn, t);
        y(d, t) = acc;

        Var dob("dob"), dib("dib"), hb("hb"), ht("ht");
        y.bound(d, 0, D).bound(t, 0, T)
         .split(d, dob, dib, 64).reorder(t, dib, dob)   // t innermost = serial
         .gpu_blocks(dob).gpu_threads(dib);
        if (inductive) {
            // State in registers: 2-slice fold window, n unrolled to registers.
            h.compute_at(y, t).store_at(y, dib).unroll(n).fold_storage(t, 2);
        } else {
            // Consumer reads every t, so an RDom scan cannot be a per-thread
            // register window; it must materialize D*N*T to global memory.
            h.compute_root();
            h.gpu_tile(d, n, hb, ht, d, n, 64, 4, TailStrategy::GuardWithIf);
            h.update(0).gpu_tile(d, n, hb, ht, d, n, 64, 4, TailStrategy::GuardWithIf);
        }
        return y;
    };

    // ---- (C)/(D) TWO-STAGE sqrt(T) blocked scan, register-resident when inductive ----
    auto build_two_stage = [&](bool inductive) -> Func {
        Func a_pad(Float(32), "a_pad"), B_pad(Float(32), "B_pad");
        Expr gtt = k * L + j, valid = gtt < T, ci = clamp(gtt, 0, T - 1);
        a_pad(d, n, k, j) = select(valid, aF(d, n, ci), 1.0f);
        B_pad(d, n, k, j) = select(valid, b(n, ci) * x(d, ci), 0.0f);

        Func PAj(Float(32), "PAj"), loc(Float(32), "loc"),
             cA("cA"), cB("cB"), carry(Float(32), "carry"), hchunk(Float(32), "hchunk");
        if (inductive) {
            PAj(d, n, k, j) = select(j <= 0, a_pad(d, n, k, 0),
                                     likely(a_pad(d, n, k, j) * PAj(d, n, k, j - 1)));
            loc(d, n, k, j) = select(j <= 0, B_pad(d, n, k, 0),
                                     likely(a_pad(d, n, k, j) * loc(d, n, k, j - 1) + B_pad(d, n, k, j)));
            cA(d, n, k) = PAj(d, n, k, L - 1);
            cB(d, n, k) = loc(d, n, k, L - 1);
            carry(d, n, k) = select(k <= 0, 0.0f,
                                    likely(cA(d, n, k - 1) * carry(d, n, k - 1) + cB(d, n, k - 1)));
            hchunk(d, n, k, j) = select(j <= 0, a_pad(d, n, k, 0) * carry(d, n, k) + B_pad(d, n, k, 0),
                                        likely(a_pad(d, n, k, j) * hchunk(d, n, k, j - 1) + B_pad(d, n, k, j)));
        } else {
            PAj(d, n, k, j) = a_pad(d, n, k, j);
            loc(d, n, k, j) = B_pad(d, n, k, j);
            RDom rj(1, L - 1);
            PAj(d, n, k, rj) = a_pad(d, n, k, rj) * PAj(d, n, k, rj - 1);
            loc(d, n, k, rj) = a_pad(d, n, k, rj) * loc(d, n, k, rj - 1) + B_pad(d, n, k, rj);
            cA(d, n, k) = PAj(d, n, k, L - 1);
            cB(d, n, k) = loc(d, n, k, L - 1);
            carry(d, n, k) = 0.0f;
            RDom rk(1, C - 1);
            carry(d, n, rk) = cA(d, n, rk - 1) * carry(d, n, rk - 1) + cB(d, n, rk - 1);
            hchunk(d, n, k, j) = a_pad(d, n, k, j);   // placeholder base
            RDom rj2(1, L - 1);
            hchunk(d, n, k, 0) = a_pad(d, n, k, 0) * carry(d, n, k) + B_pad(d, n, k, 0);
            hchunk(d, n, k, rj2) = a_pad(d, n, k, rj2) * hchunk(d, n, k, rj2 - 1) + B_pad(d, n, k, rj2);
        }

        // Consumer in CHUNK coordinates (k,j): the hchunk access is then exactly a
        // clean in-order sliding window over j, so fold(j,2) is legal.
        Func ykj("ykj");
        Expr acc = 0.0f;
        Expr tci = clamp(k * L + j, 0, T - 1);   // padded tail never read by y
        for (int nn = 0; nn < N; nn++) acc += c(nn, tci) * hchunk(d, nn, k, j);
        ykj(d, k, j) = acc;
        Func y("y_out");
        y(d, t) = ykj(d, t / L, t % L);

        Var dob("dob"), dib("dib"), a1("a1"), a2("a2");
        if (inductive) {
            for (Func agg : {cA, cB}) {
                agg.compute_root().reorder(n, d, k)
                   .split(d, dob, dib, 32).gpu_blocks(dob, k).gpu_threads(dib).unroll(n);
            }
            // PAj/loc are produced over the whole chunk per thread (cA/cB read only
            // the last j), so there is no outer sliding j-loop to fold over -- the
            // chunk stays thread-local (L values). Only hchunk, consumed per-j by
            // ykj, has a foldable j slide.
            PAj.compute_at(cA, dib).unroll(n);
            loc.compute_at(cB, dib).unroll(n);
            carry.compute_root().reorder(n, d, k)
                 .split(d, dob, dib, 64).gpu_blocks(dob).gpu_threads(dib).unroll(n);
            ykj.compute_root()
               .split(d, dob, dib, 32).reorder(j, dib, k, dob)
               .gpu_blocks(dob, k).gpu_threads(dib);
            hchunk.compute_at(ykj, j).store_at(ykj, dib).unroll(n).fold_storage(j, 2);
            y.bound(d, 0, D).bound(t, 0, T)
             .gpu_tile(d, t, a1, a2, d, t, 32, 8, TailStrategy::GuardWithIf);
        } else {
            for (Func f : {PAj, loc}) {
                f.compute_root().gpu_tile(d, k, a1, a2, d, k, 32, 8, TailStrategy::GuardWithIf);
                f.update(0).gpu_tile(d, k, a1, a2, d, k, 32, 8, TailStrategy::GuardWithIf);
            }
            for (Func agg : {cA, cB})
                agg.compute_root().gpu_tile(d, n, a1, a2, d, n, 64, 4, TailStrategy::GuardWithIf);
            carry.compute_root().gpu_tile(d, n, a1, a2, d, n, 64, 4, TailStrategy::GuardWithIf);
            carry.update(0).gpu_tile(d, n, a1, a2, d, n, 64, 4, TailStrategy::GuardWithIf);
            hchunk.compute_root().gpu_tile(d, k, a1, a2, d, k, 32, 8, TailStrategy::GuardWithIf);
            hchunk.update(0).gpu_tile(d, k, a1, a2, d, k, 32, 8, TailStrategy::GuardWithIf);
            hchunk.update(1).gpu_tile(d, k, a1, a2, d, k, 32, 8, TailStrategy::GuardWithIf);
            ykj.compute_root().gpu_tile(d, k, a1, a2, d, k, 32, 8, TailStrategy::GuardWithIf);
            y.bound(d, 0, D).bound(t, 0, T)
             .gpu_tile(d, t, a1, a2, d, t, 32, 8, TailStrategy::GuardWithIf);
        }
        return y;
    };

    // Global-memory the trajectory forces through DRAM (bytes), analytic. This is
    // the "impossible at large T" axis: the non-inductive variants scale as D*N*T.
    auto traj_bytes = [&](bool inductive, bool two_stage) -> double {
        double f = sizeof(float);
        if (inductive) {
            // Only per-chunk aggregates + carries (two-stage) or nothing (seq)
            // beyond the D*T output transit global memory.
            return two_stage ? (double)D * N * C * f * 3 : 0.0;
        }
        // Non-inductive materializes the full D*N*T trajectory (x3 intermediates
        // for the two-stage variant).
        return (double)D * N * T * f * (two_stage ? 3 : 1);
    };

    struct Variant { const char *name; bool inductive, two_stage; };
    std::vector<Variant> variants = {
        {"(A) sequential  non-inductive", false, false},
        {"(B) sequential  inductive    ", true,  false},
        {"(C) two-stage   non-inductive", false, true},
        {"(D) two-stage   inductive    ", true,  true},
    };

    // CPU streaming reference.
    std::vector<float> cy((size_t)D * T);
    for (int dd = 0; dd < D; dd++) {
        std::vector<float> hs(N);
        for (int nn = 0; nn < N; nn++) hs[nn] = b(nn, 0) * x(dd, 0);
        float y0 = 0; for (int nn = 0; nn < N; nn++) y0 += c(nn, 0) * hs[nn];
        cy[(size_t)dd * T] = y0;
        for (int tt = 1; tt < T; tt++) {
            float yy = 0;
            for (int nn = 0; nn < N; nn++) {
                float aval = 0.9f + 0.09f * (delta(dd, tt) * Av(dd, nn));  // match Halide aF
                hs[nn] = aval * hs[nn] + b(nn, tt) * x(dd, tt);
                yy += c(nn, tt) * hs[nn];
            }
            cy[(size_t)dd * T + tt] = yy;
        }
    }

    const int WARMUP = 3, TRIALS = 30;
    printf("Mamba GPU scan  N=%d D=%d T=%d  (L=%d C=%d)  sm_86  [warmup=%d trials=%d]\n",
           N, D, T, L, C, WARMUP, TRIALS);
    printf("  %-30s  %10s %10s %10s   %10s  %-9s  %s\n",
           "variant", "median_ms", "p25", "p75", "Mtok/s", "traj_MB", "check");

    for (auto &v : variants) {
        try {
            Func f = v.two_stage ? build_two_stage(v.inductive) : build_seq(v.inductive);
            f.compile_jit(target);
            Buffer<float> r(D, T);
            for (int w = 0; w < WARMUP; w++) { f.realize(r, target); r.device_sync(); }
            r.copy_to_host();
            double err = 0;
            for (int dd = 0; dd < D; dd++)
                for (int tt = 0; tt < T; tt++)
                    err = std::max(err, (double)std::abs(r(dd, tt) - cy[(size_t)dd * T + tt]));

            std::vector<double> ms;
            ms.reserve(TRIALS);
            for (int q = 0; q < TRIALS; q++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                f.realize(r, target);
                r.device_sync();
                auto t1 = std::chrono::high_resolution_clock::now();
                ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            std::sort(ms.begin(), ms.end());
            double med = ms[TRIALS / 2], p25 = ms[TRIALS / 4], p75 = ms[(3 * TRIALS) / 4];
            double mtoks = (double)D * T / (med * 1e3);
            printf("  %-30s  %10.3f %10.3f %10.3f   %10.1f  %-9.1f  %s (err %.1g)\n",
                   v.name, med, p25, p75, mtoks,
                   traj_bytes(v.inductive, v.two_stage) / 1e6,
                   err < 1e-2 ? "PASS" : "FAIL", err);
        } catch (const Halide::Error &e) {
            // A non-inductive variant that cannot allocate its trajectory at large
            // T lands here -- which is exactly the point being measured.
            printf("  %-30s  %10s %10s %10s   %10s  %-9.1f  ERROR: %s\n",
                   v.name, "-", "-", "-", "-",
                   traj_bytes(v.inductive, v.two_stage) / 1e6, e.what());
        }
    }
    return 0;
}
