// Register/shared-resident TWO-STAGE GPU scan: the schedule that has BOTH halves
// the fused mamba_ssm kernel has -- register-resident state AND sqrt(T) chunk
// parallelism -- with NO global materialization of the per-position trajectory.
//   Kernel 1 (parallel over d,k): each thread scans its chunk in registers,
//            emits only the small chunk aggregates cA=prod(a), cB=local end state.
//   Kernel 2 (parallel over d,n): short serial scan of the C chunk carries.
//   Kernel 3 (parallel over d,k): reload carry, RE-scan the chunk in registers
//            (carry folded into the chunk's initial condition), emit y in order.
// Only D*N*C aggregates/carries transit global memory; the D*N*T trajectory never
// does. Parallelism = D*C threads (fixes the sequential-scan occupancy problem).
// Compares inductive (register j-fold) vs non-inductive (must materialize).
//
// Build: g++ apps/mamba/mamba_gpu_fused2.cpp -O3 -march=native -Iinclude
//   -Lbuild/src -lHalide -lpthread -ldl -o /tmp/mgf2 -std=c++17
//   LD_LIBRARY_PATH=build/src:/usr/lib/wsl/lib /tmp/mgf2 [N D T]

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 16;
    int D = argc > 2 ? atoi(argv[2]) : 512;
    int T = argc > 3 ? atoi(argv[3]) : 16384;
    int L = (int)std::ceil(std::sqrt((double)T));
    int C = (T + L - 1) / L;

    Target target = get_host_target().with_feature(Target::CUDA).with_feature(Target::CUDACapability86);

    Buffer<float> a(D, N, T), b(N, T), c(N, T), x(D, T);
    srand(5);
    for (int t = 0; t < T; t++) {
        for (int n = 0; n < N; n++) {
            b(n, t) = (rand() % 200) / 100.0f - 1.0f;
            c(n, t) = (rand() % 200) / 100.0f - 1.0f;
            for (int d = 0; d < D; d++)
                a(d, n, t) = 0.9f + (rand() % 99) / 1000.0f;
        }
        for (int d = 0; d < D; d++)
            x(d, t) = (rand() % 200) / 100.0f - 1.0f;
    }

    Var d("d"), n("n"), k("k"), j("j"), t("t");

    auto build = [&](bool inductive) -> Func {
        Func a_pad(Float(32), "a_pad"), B_pad(Float(32), "B_pad");
        Expr gtt = k * L + j, valid = gtt < T, ci = clamp(gtt, 0, T - 1);
        a_pad(d, n, k, j) = select(valid, a(d, n, ci), 1.0f);
        B_pad(d, n, k, j) = select(valid, b(n, ci) * x(d, ci), 0.0f);

        Func PAj(Float(32), "PAj"), loc(Float(32), "loc"),
            cA("cA"), cB("cB"), carry(Float(32), "carry"),
            hchunk(Float(32), "hchunk");
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
            hchunk(d, n, k, j) = a_pad(d, n, k, j);  // placeholder base
            RDom rj2(1, L - 1);
            hchunk(d, n, k, 0) = a_pad(d, n, k, 0) * carry(d, n, k) + B_pad(d, n, k, 0);
            hchunk(d, n, k, rj2) = a_pad(d, n, k, rj2) * hchunk(d, n, k, rj2 - 1) + B_pad(d, n, k, rj2);
        }

        // Consumer in CHUNK coordinates (k,j): the hchunk access is then exactly
        // (k,j) -> a clean in-order sliding window over j, so fold(j,2) is legal.
        // (Indexing by t%L hid the monotonic access from the fold analysis.)
        Func ykj("ykj");
        Expr acc = 0.0f;
        Expr tci = clamp(k * L + j, 0, T - 1);  // padded tail (C*L>T) never read by y
        for (int nn = 0; nn < N; nn++)
            acc += c(nn, tci) * hchunk(d, nn, k, j);
        ykj(d, k, j) = acc;
        Func y("y_out");
        y(d, t) = ykj(d, t / L, t % L);

        Var dob("dob"), dib("dib"), a1, a2;
        if (inductive) {
            // Aggregates cA,cB: parallel over (d,k); internal j-scan register-folded.
            for (Func agg : {cA, cB}) {
                agg.compute_root().reorder(n, d, k).split(d, dob, dib, 32).gpu_blocks(dob, k).gpu_threads(dib).unroll(n);
            }
            PAj.compute_at(cA, dib).unroll(n).fold_storage(j, 2);
            loc.compute_at(cB, dib).unroll(n).fold_storage(j, 2);
            // Carry: parallel over (d,n), short serial scan over k.
            carry.compute_root().reorder(n, d, k).split(d, dob, dib, 64).gpu_blocks(dob).gpu_threads(dib).unroll(n);
            // Output: parallel over (d,k); per-chunk register j-scan, emit in order.
            ykj.compute_root()
                .split(d, dob, dib, 32)
                .reorder(j, dib, k, dob)
                .gpu_blocks(dob, k)
                .gpu_threads(dib);
            hchunk.compute_at(ykj, j).store_at(ykj, dib).unroll(n).fold_storage(j, 2);
            y.bound(d, 0, D).bound(t, 0, T).gpu_tile(d, t, a1, a2, d, t, 32, 8, TailStrategy::GuardWithIf);
        } else {
            // Non-inductive: every scan materializes to global (RDom over j/k), and
            // the y consumer reads all j -> hchunk cannot fold. compute_root all.
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
            y.bound(d, 0, D).bound(t, 0, T).gpu_tile(d, t, a1, a2, d, t, 32, 8, TailStrategy::GuardWithIf);
        }
        return y;
    };

    try {
        std::vector<float> cy((size_t)D * T);
        for (int dd = 0; dd < D; dd++) {
            std::vector<float> hs(N);
            for (int nn = 0; nn < N; nn++)
                hs[nn] = b(nn, 0) * x(dd, 0);
            float y0 = 0;
            for (int nn = 0; nn < N; nn++)
                y0 += c(nn, 0) * hs[nn];
            cy[(size_t)dd * T] = y0;
            for (int tt = 1; tt < T; tt++) {
                float yy = 0;
                for (int nn = 0; nn < N; nn++) {
                    hs[nn] = a(dd, nn, tt) * hs[nn] + b(nn, tt) * x(dd, tt);
                    yy += c(nn, tt) * hs[nn];
                }
                cy[(size_t)dd * T + tt] = yy;
            }
        }

        printf("Mamba GPU register-resident two-stage  N=%d D=%d T=%d (L=%d C=%d) sm_89\n", N, D, T, L, C);
        const char *names[2] = {"non-inductive (materialize) ", "inductive (register 2-stage)"};
        for (int mode = 0; mode < 2; mode++) {
            Func f = build(mode == 1);
            f.compile_jit(target);
            Buffer<float> r(D, T);
            f.realize(r, target);
            r.copy_to_host();
            double err = 0;
            for (int dd = 0; dd < D; dd++)
                for (int tt = 0; tt < T; tt++)
                    err = std::max(err, (double)std::abs(r(dd, tt) - cy[(size_t)dd * T + tt]));
            double best = 1e18;
            for (int q = 0; q < 5; q++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                f.realize(r, target);
                r.device_sync();
                auto t1 = std::chrono::high_resolution_clock::now();
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            printf("  %s  %8.3f ms   err %.2g %s\n", names[mode], best, err,
                   err < 1e-2 ? "PASS" : "FAIL");
        }
        return 0;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
