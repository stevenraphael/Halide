// Batched Mamba GPU sequential scan: does adding a BATCH axis fix the occupancy
// problem of the register-resident sequential-inductive scan?
//
// In the single-sequence bench (mamba_gpu_bench.cpp) the sequential-inductive
// kernel (B) keeps state in registers but its only parallel axis is D (~512
// lanes) -> the GPU is starved and it loses badly. The production selective_scan
// kernel avoids this by gridding over batch*dim: many independent sequences give
// it thousands of parallel lanes while each lane still runs a serial, register-
// resident scan over t. This test reproduces that: parallel lanes = batch*D, t
// serial per lane, state folded to a 2-slice register window. We sweep the batch
// size and show throughput scaling up as occupancy fills.
//
//   inductive     : h.compute_at(y,t).store_at(y,thread).unroll(n).fold_storage(t,2)
//                   parallel over (batch, d); t serial inside the kernel.
//   non-inductive : RDom-over-t scan; consumer reads all t so h cannot be a
//                   per-lane register window -> materializes O(batch*D*N*T).
//
// Decay uses the real Mamba parameterization (A time-invariant (D,N), delta
// (D,T,batch)); deltaA is recomputed inline, so inputs are O(batch*D*T), not
// O(batch*D*N*T).
//
// Build: g++ apps/mamba/mamba_gpu_batched.cpp -O3 -march=native
//   -Idistrib_install/include -Ldistrib_install/lib -lHalide -lpthread -ldl
//   -Wl,-rpath,distrib_install/lib -o /tmp/mgbat -std=c++17
//   LD_LIBRARY_PATH=distrib_install/lib:/usr/lib/wsl/lib /tmp/mgbat [N D T]

#include "../support/bench_harness.h"
#include "Halide.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 16;
    int D = argc > 2 ? atoi(argv[2]) : 512;
    int T = argc > 3 ? atoi(argv[3]) : 4096;

    Target target = get_host_target().with_feature(Target::CUDA).with_feature(Target::CUDACapability86);

    hb::print_spec_header("mamba_gpu_batched", target.to_string(),
                          "sequential scan; parallel lanes = batch*D; t serial per lane");

    // A is shared across the batch (per channel-state, time-invariant).
    Buffer<float> Av(D, N);
    srand(5);
    for (int dd = 0; dd < D; dd++)
        for (int nn = 0; nn < N; nn++)
            Av(dd, nn) = -(0.01f + (rand() % 99) / 100.0f);

    Var d("d"), n("n"), t("t"), bb("bb");
    // Affine stable decay computed inline from the small inputs (matches the
    // single-sequence bench): deltaA ~ 0.9 + 0.09*delta*A, delta>0, A<0.
    // delta is per (d,t,batch); A shared.
    auto make_aF = [&](Buffer<float> &delta) {
        return [&delta, &Av](const Expr &dd, const Expr &nn, const Expr &tt,
                             const Expr &b_) -> Expr {
            return exp(delta(dd, tt, b_) * Av(dd, nn));  // real Mamba deltaA=exp(delta*A)
        };
    };

    auto run_for_batch = [&](int Bsz) {
        // Per-batch inputs. O(batch*D*T) for delta/x, O(batch*N*T) for b/c.
        Buffer<float> delta(D, T, Bsz), x(D, T, Bsz), b(N, T, Bsz), c(N, T, Bsz);
        for (int q = 0; q < Bsz; q++) {
            for (int tt = 0; tt < T; tt++) {
                for (int nn = 0; nn < N; nn++) {
                    b(nn, tt, q) = (rand() % 200) / 100.0f - 1.0f;
                    c(nn, tt, q) = (rand() % 200) / 100.0f - 1.0f;
                }
                for (int dd = 0; dd < D; dd++) {
                    delta(dd, tt, q) = (rand() % 100) / 1000.0f;
                    x(dd, tt, q) = (rand() % 200) / 100.0f - 1.0f;
                }
            }
        }
        auto aF = make_aF(delta);

        auto build = [&](bool inductive) -> Func {
            Func h(Float(32), "h");
            // Cache the time-invariant A(d,n) so it is loaded ONCE per lane (outside
            // the t-loop) instead of re-read every timestep. delta(d,t) still streams.
            Func avc(Float(32), "avc");
            avc(d, n) = Av(d, n);
            auto decay = [&](const Expr &dd, const Expr &nn, const Expr &tt, const Expr &b_) {
                return exp(delta(dd, tt, b_) * avc(dd, nn));  // real Mamba deltaA=exp(delta*A)
            };
            // Match v1's arithmetic exactly: the input term is the discretized
            // deltaB*u = delta * B * u (an extra delta multiply on b*x), not b*x.
            auto dBu = [&](const Expr &dd, const Expr &nn, const Expr &tt, const Expr &b_) {
                return delta(dd, tt, b_) * b(nn, tt, b_) * x(dd, tt, b_);
            };
            if (inductive) {
                h(d, n, t, bb) = select(t <= 0, dBu(d, n, 0, bb),
                                        likely(decay(d, n, t, bb) * h(d, n, t - 1, bb) +
                                               dBu(d, n, t, bb)));
            } else {
                h(d, n, t, bb) = dBu(d, n, 0, bb);
                RDom rt(1, T - 1);
                h(d, n, rt, bb) = aF(d, n, rt, bb) * h(d, n, rt - 1, bb) +
                                  dBu(d, n, rt, bb);
                // Materialized scan (schedule here, where rt is in scope): parallel
                // over (d, bb); time rt is the serial scan placed OUTERMOST (one
                // kernel per step), so no serial loop sits between GPU block loops.
                Var dob2("dob2"), dib2("dib2");
                h.compute_root();
                h.split(d, dob2, dib2, 64).reorder(t, n, dib2, dob2, bb).gpu_blocks(dob2, bb).gpu_threads(dib2);
                h.update(0).split(d, dob2, dib2, 64).reorder(n, dib2, dob2, bb, rt).gpu_blocks(dob2, bb).gpu_threads(dib2);
            }
            Func y("y_out");
            Expr acc = 0.0f;
            for (int nn = 0; nn < N; nn++)
                acc += c(nn, t, bb) * h(d, nn, t, bb);
            y(d, t, bb) = acc;

            Var dob("dob"), dib("dib"), hb_("hb"), ht_("ht");
            y.bound(d, 0, D).bound(t, 0, T).bound(bb, 0, Bsz).split(d, dob, dib, 64).reorder(t, dib, dob, bb)  // t serial innermost
                .gpu_blocks(dob, bb)
                .gpu_threads(dib);  // lanes = batch * (D/64)*64
            if (inductive) {
                h.compute_at(y, t).store_at(y, dib).unroll(n).fold_storage(t, hb::fold_factor(1, T));
                // Load A(d,n) once per lane (at the thread level, outside the t-loop).
                avc.compute_at(y, dib).unroll(n);
            }
            // (non-inductive h is scheduled at its definition, where rt is in scope)
            return y;
        };

        // CPU streaming reference. O(batch*D*N*T) -- ruinously slow at realistic
        // dims (D>=2048), so it is opt-in via CHECK=1; the reported timing is
        // unaffected either way. When skipped, err is reported as -1 (unchecked).
        const bool do_check = getenv("CHECK") != nullptr;
        std::vector<float> cy(do_check ? (size_t)Bsz * D * T : 0);
        if (do_check)
            for (int q = 0; q < Bsz; q++)
                for (int dd = 0; dd < D; dd++) {
                    std::vector<float> hs(N);
                    for (int nn = 0; nn < N; nn++)
                        hs[nn] = delta(dd, 0, q) * b(nn, 0, q) * x(dd, 0, q);
                    float y0 = 0;
                    for (int nn = 0; nn < N; nn++)
                        y0 += c(nn, 0, q) * hs[nn];
                    cy[((size_t)q * D + dd) * T] = y0;
                    for (int tt = 1; tt < T; tt++) {
                        float yy = 0;
                        for (int nn = 0; nn < N; nn++) {
                            float av = std::exp(delta(dd, tt, q) * Av(dd, nn));
                            hs[nn] = av * hs[nn] + delta(dd, tt, q) * b(nn, tt, q) * x(dd, tt, q);
                            yy += c(nn, tt, q) * hs[nn];
                        }
                        cy[((size_t)q * D + dd) * T + tt] = yy;
                    }
                }

        const double toks = (double)Bsz * D * T;
        char label[64];
        // Non-inductive (materialize) does T serial kernel launches -- very slow at
        // large T -- so skip it by default; set MAMBA_NONIND=1 to include it.
        int mode0 = getenv("MAMBA_NONIND") ? 0 : 1;
        for (int mode = mode0; mode < 2; mode++) {
            bool inductive = (mode == 1);
            snprintf(label, sizeof(label), "B=%-3d %s", Bsz,
                     inductive ? "inductive (reg fold)" : "non-inductive (mat) ");
            try {
                Func f = build(inductive);
                f.compile_jit(target);
                Buffer<float> r(D, T, Bsz);
                f.realize(r, target);
                r.copy_to_host();
                double err = -1.0;
                if (do_check) {
                    err = 0;
                    for (int q = 0; q < Bsz; q++)
                        for (int dd = 0; dd < D; dd++)
                            for (int tt = 0; tt < T; tt++)
                                err = std::max(err, (double)std::abs(
                                                        r(dd, tt, q) - cy[((size_t)q * D + dd) * T + tt]));
                }
                hb::Stats s = hb::bench([&] { f.realize(r, target); r.device_sync(); });
                double state = inductive ? 0.0 : (double)Bsz * D * N * T * sizeof(float);
                double mtoks = toks / (s.min * 1e3);
                hb::print_row(label, s, mtoks, "Mtok/s", state, err, !do_check || err < 1e-2,
                              inductive ? "win" : "");
            } catch (const Halide::Error &e) {
                printf("  %-30s  ERROR: %s\n", label, e.what());
            }
        }
    };

    // Optional 4th arg: run a single batch size instead of the full sweep.
    std::vector<int> batches = (argc > 4) ? std::vector<int>{atoi(argv[4])} : std::vector<int>{1, 2, 4, 8, 16};
    for (int Bsz : batches)
        run_for_batch(Bsz);
    return 0;
}
