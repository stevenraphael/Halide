// GPU comparison for the Mamba selective-SSM scan (diagonal recurrence
//   h_t = a_t h_{t-1} + b_t x_t ,  y_t = sum_n c_t h_t ).  Four kernels:
//   (A) SEQUENTIAL scan, non-inductive (RDom over t; serial critical path = T)
//   (B) SEQUENTIAL scan, inductive
//   (C) TWO-STAGE sqrt(T) blocked scan, non-inductive (critical path ~ 2*sqrt(T))
//   (D) TWO-STAGE sqrt(T) blocked scan, inductive
// On GPU the lane count is D*N; sequential serializes T steps, the blocked scan
// serializes only the C~sqrt(T) inter-chunk carries while the intra-chunk work is
// parallel across chunks -- so we expect the two-stage scan to win at long T.
//
// Build: g++ apps/mamba/mamba_gpu.cpp -O3 -march=native -Iinclude -Lbuild/src
//        -lHalide -lpthread -ldl -o /tmp/mgpu -std=c++17
//   LD_LIBRARY_PATH=build/src:/usr/lib/wsl/lib /tmp/mgpu [N D T]

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

    auto make_consumer = [&](Func h_kj, bool blocked) -> Func {
        // y(d,t) = sum_n c(n,t) h(d,n,t).  For the blocked scan, h is indexed
        // (d,n,k,j) with t = k*L+j.
        Func y("y_out");
        Expr acc = 0.0f;
        for (int nn = 0; nn < N; nn++) {
            Expr hv = blocked ? h_kj(d, nn, t / L, t % L) : h_kj(d, nn, t);
            acc += c(nn, t) * hv;
        }
        y(d, t) = acc;
        return y;
    };

    // ---- SEQUENTIAL scan ----
    auto build_seq = [&](bool inductive) -> Func {
        Func h(Float(32), "h_seq");
        if (inductive) {
            h(d, n, t) = select(t <= 0, b(n, 0) * x(d, 0),
                                likely(a(d, n, t) * h(d, n, t - 1) + b(n, t) * x(d, t)));
        } else {
            h(d, n, t) = b(n, 0) * x(d, 0);
            RDom rt(1, T - 1);
            h(d, n, rt) = a(d, n, rt) * h(d, n, rt - 1) + b(n, rt) * x(d, rt);
        }
        Func y = make_consumer(h, false);
        // GPU: lanes = (d,n) parallel; t is the serial scan axis (kernel per step).
        Var db("db"), dt("dt");
        h.compute_root();
        h.gpu_tile(d, n, db, dt, d, n, 64, 4, TailStrategy::GuardWithIf);
        if (!inductive) h.update(0).gpu_tile(d, n, db, dt, d, n, 64, 4, TailStrategy::GuardWithIf);
        y.gpu_tile(d, t, db, dt, d, t, 32, 8, TailStrategy::GuardWithIf);
        return y;
    };

    // ---- TWO-STAGE sqrt(T) blocked scan ----
    auto build_blocked = [&](bool inductive) -> Func {
        Func a_pad(Float(32), "a_pad"), B_pad(Float(32), "B_pad");
        Expr gtt = k * L + j, valid = gtt < T, ci = clamp(gtt, 0, T - 1);
        a_pad(d, n, k, j) = select(valid, a(d, n, ci), 1.0f);
        B_pad(d, n, k, j) = select(valid, b(n, ci) * x(d, ci), 0.0f);

        Func PA(Float(32), "PA"), PB(Float(32), "PB"), carry(Float(32), "carry"), h("h_blk");
        if (inductive) {
            PA(d, n, k, j) = select(j <= 0, a_pad(d, n, k, 0),
                                    likely(a_pad(d, n, k, j) * PA(d, n, k, j - 1)));
            PB(d, n, k, j) = select(j <= 0, B_pad(d, n, k, 0),
                                    likely(a_pad(d, n, k, j) * PB(d, n, k, j - 1) + B_pad(d, n, k, j)));
            carry(d, n, k) = select(k <= 0, 0.0f,
                                    likely(PA(d, n, k - 1, L - 1) * carry(d, n, k - 1) + PB(d, n, k - 1, L - 1)));
        } else {
            PA(d, n, k, j) = a_pad(d, n, k, j);
            PB(d, n, k, j) = B_pad(d, n, k, j);
            RDom rj(1, L - 1);
            PA(d, n, k, rj) = a_pad(d, n, k, rj) * PA(d, n, k, rj - 1);
            PB(d, n, k, rj) = a_pad(d, n, k, rj) * PB(d, n, k, rj - 1) + B_pad(d, n, k, rj);
            carry(d, n, k) = 0.0f;
            RDom rk(1, C - 1);
            carry(d, n, rk) = PA(d, n, rk - 1, L - 1) * carry(d, n, rk - 1) + PB(d, n, rk - 1, L - 1);
        }
        h(d, n, k, j) = PA(d, n, k, j) * carry(d, n, k) + PB(d, n, k, j);
        Func y = make_consumer(h, true);

        Var db("db"), dt("dt");
        // Stage 1: parallel across (d, chunk k); j is the serial intra-chunk axis.
        for (Func f : {PA, PB}) {
            f.compute_root().gpu_tile(d, k, db, dt, d, k, 32, 8, TailStrategy::GuardWithIf);
            if (!inductive) f.update(0).gpu_tile(d, k, db, dt, d, k, 32, 8, TailStrategy::GuardWithIf);
        }
        // Stage 2: parallel across (d,n); k is the (short) serial carry axis.
        carry.compute_root().gpu_tile(d, n, db, dt, d, n, 64, 4, TailStrategy::GuardWithIf);
        if (!inductive) carry.update(0).gpu_tile(d, n, db, dt, d, n, 64, 4, TailStrategy::GuardWithIf);
        h.compute_root().gpu_tile(d, k, db, dt, d, k, 32, 8, TailStrategy::GuardWithIf);
        y.gpu_tile(d, t, db, dt, d, t, 32, 8, TailStrategy::GuardWithIf);
        return y;
    };

    try {
        struct K {
            const char *name;
            Func f;
        };
        std::vector<K> ks = {
            {"(A) sequential  non-inductive", build_seq(false)},
            {"(B) sequential  inductive    ", build_seq(true)},
            {"(C) two-stage   non-inductive", build_blocked(false)},
            {"(D) two-stage   inductive    ", build_blocked(true)},
        };
        // CPU reference.
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

        printf("Mamba GPU scan  N=%d D=%d T=%d  (L=%d C=%d)  sm_89\n", N, D, T, L, C);
        for (auto &kern : ks) {
            kern.f.compile_jit(target);
            Buffer<float> r(D, T);
            kern.f.realize(r, target);  // warm
            r.copy_to_host();
            double err = 0;
            for (int dd = 0; dd < D; dd++)
                for (int tt = 0; tt < T; tt++)
                    err = std::max(err, (double)std::abs(r(dd, tt) - cy[(size_t)dd * T + tt]));
            double best = 1e18;
            for (int q = 0; q < 5; q++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                kern.f.realize(r, target);
                r.device_sync();
                auto t1 = std::chrono::high_resolution_clock::now();
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            printf("  %s  %8.3f ms   err %.2g %s\n", kern.name, best, err,
                   err < 1e-2 ? "PASS" : "FAIL");
        }
        return 0;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
