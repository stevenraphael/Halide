// Fully-fused GPU schedule for the sequential Mamba scan: keep the state h in
// REGISTERS per GPU thread (one thread per channel d), serial t-loop INSIDE the
// kernel, no global-memory materialization of the trajectory. Tests how much of
// the 17x gap to the fused mamba_ssm CUDA kernel is pure global-mem round-trip.
//   inductive:     h.compute_at(y,t).store_at(y,thread).fold(t,2).unroll(n)
//                  -> sliding 2-slice window in registers, recompute-free.
//   non-inductive: an RDom-over-t scan's consumer reads every t, so h cannot be a
//                  register sliding window (would be O(T^2) recompute at thread
//                  level); it must materialize -> compute_root global buffer.
//                  This asymmetry is the whole point.
//
// Build: g++ apps/mamba/mamba_gpu_fused.cpp -O3 -march=native -Iinclude
//   -Lbuild/src -lHalide -lpthread -ldl -o /tmp/mgf -std=c++17
//   LD_LIBRARY_PATH=build/src:/usr/lib/wsl/lib /tmp/mgf [N D T]

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

    Var d("d"), n("n"), t("t"), dob("dob"), dib("dib");

    auto build = [&](bool inductive) -> Func {
        Func h(Float(32), "h");
        if (inductive) {
            h(d, n, t) = select(t <= 0, b(n, 0) * x(d, 0),
                                likely(a(d, n, t) * h(d, n, t - 1) + b(n, t) * x(d, t)));
        } else {
            h(d, n, t) = b(n, 0) * x(d, 0);
            RDom rt(1, T - 1);
            h(d, n, rt) = a(d, n, rt) * h(d, n, rt - 1) + b(n, rt) * x(d, rt);
        }
        Func y("y_out");
        Expr acc = 0.0f;
        for (int nn = 0; nn < N; nn++)
            acc += c(nn, t) * h(d, nn, t);
        y(d, t) = acc;

        // One GPU thread per channel d; serial t-loop INSIDE the kernel.
        y.bound(d, 0, D).bound(t, 0, T).split(d, dob, dib, 64).reorder(t, dib, dob)  // t innermost (serial), then thread, block
            .gpu_blocks(dob)
            .gpu_threads(dib);
        if (inductive) {
            // State lives in registers: computed each step, 2-slice fold window,
            // n unrolled so the 2*N values are registers, not local memory.
            h.compute_at(y, t).store_at(y, dib).unroll(n).fold_storage(t, 2);
        } else {
            // Must materialize: consumer reads all t, RDom scan can't fold into a
            // per-thread register window. Global buffer round-trip.
            h.compute_root();
            Var hb, ht;
            h.gpu_tile(d, n, hb, ht, d, n, 64, 4, TailStrategy::GuardWithIf);
            h.update(0).gpu_tile(d, n, hb, ht, d, n, 64, 4, TailStrategy::GuardWithIf);
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

        printf("Mamba GPU FUSED (register-resident) scan  N=%d D=%d T=%d  sm_89\n", N, D, T);
        const char *names[2] = {"non-inductive (materialize)", "inductive (register fold) "};
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
