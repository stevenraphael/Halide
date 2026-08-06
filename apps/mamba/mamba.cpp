// Mamba selective-SSM core scan, arbitrary state dim N, per-step consumer.
//   h_{d,n,t} = a_{d,n,t} * h_{d,n,t-1} + b_{n,t} * x_{d,t}   (DIAGONAL recurrence)
//   y_{d,t}   = sum_n c_{n,t} * h_{d,n,t}                     (per-step output)
// D channels (the wide independent/vectorize axis), N state (small, ~16), T tokens.
// a,b,c,x are input-dependent (the "selective" part) -> supplied as input buffers.
//
// Diagonal recurrence => the ONLY shifted dim is t. So the non-inductive scan's
// recursing dim is naturally an RVar (the time RDom); n,d stay pure vars but the
// self-call reads them UNSHIFTED (matches) so the pure-var-only inductive check
// correctly classifies it non-inductive. Cleaner than kalman_ss (no matvec, so no
// need to force the output index to an RVar). Inductive folds the h trajectory
// (N*D*T) to a 2-slice window fused into y; win shows up when N*D*T spills cache.
//
// Build: g++ apps/mamba/mamba.cpp -O3 -march=native -Iinclude -Lbuild/src
//        -lHalide -lpthread -ldl -o /tmp/mamba -std=c++17
//        LD_LIBRARY_PATH=build/src /tmp/mamba [N D T]

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 16;    // state dim (real Mamba: 16)
    int D = argc > 2 ? atoi(argv[2]) : 4096;  // channels (wide, vectorize/parallel)
    int T = argc > 3 ? atoi(argv[3]) : 512;   // tokens (scan length)

    // Inputs. a in (0.9,0.999) for a stable decaying recurrence; b,c,x random.
    // Layouts chosen so the vectorized channel axis d is innermost/contiguous.
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

    auto build = [&](bool inductive) -> Func {
        Var d("d"), n("n"), t("t");
        Func h(Float(32), "h");
        if (inductive) {
            // Scan axis t is a Var -> self-call h(d,n,t-1) at a pure-var shift is a
            // genuine inductive self-reference (diagonal: same d,n, only t shifts).
            h(d, n, t) = select(t <= 0, b(n, 0) * x(d, 0),
                                likely(a(d, n, t) * h(d, n, t - 1) + b(n, t) * x(d, t)));
        } else {
            // FAIR non-inductive: identical recurrence as an explicit time RDom scan.
            // t-position is the RVar rt (LHS reduction var) so the pure-var check
            // skips it; d,n are pure vars but the self-call reads them UNSHIFTED
            // (matches) -> correctly NOT inductive. Only difference vs inductive is
            // materialize-vs-fold.
            h(d, n, t) = b(n, 0) * x(d, 0);  // t=0 init (pure def)
            RDom rt(1, T - 1);               // time scan
            h(d, n, rt) = a(d, n, rt) * h(d, n, rt - 1) + b(n, rt) * x(d, rt);
        }
        // Per-step consumer y(d,t) = sum_n c(n,t) h(d,n,t). Manual unroll over the
        // small state dim n (pure func, constant n) so inductive h fuses into it.
        Func y("y");
        Expr acc = 0.0f;
        for (int nn = 0; nn < N; nn++)
            acc += c(nn, t) * h(d, nn, t);
        y(d, t) = acc;
        // Channel d is the parallel + vectorizable axis; t is the serial scan axis.
        const int V = 8;
        Var do_("do"), di("di");
        y.bound(d, 0, D).bound(t, 0, T).split(d, do_, di, V).reorder(di, t, do_).vectorize(di).parallel(do_);
        // h stored d-innermost so vectorizing over d is a contiguous SIMD load.
        if (inductive) {
            h.reorder_storage(d, n, t)
                .compute_at(y, t)
                .store_at(y, do_)
                .fold_storage(t, 2)
                .vectorize(d, V);
        } else {
            h.reorder_storage(d, n, t).compute_at(y, do_).vectorize(d, V);
            h.update(0).vectorize(d, V);
        }
        return y;
    };

    try {
        Func yi = build(true), yn = build(false);
        yi.compile_jit();
        yn.compile_jit();
        Buffer<float> ri(D, T), rn(D, T);
        yi.realize(ri);
        yn.realize(rn);

        auto bench = [&](Func f, Buffer<float> &bb) {
            double best = 1e18;
            for (int k = 0; k < 5; k++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                f.realize(bb);
                auto t1 = std::chrono::high_resolution_clock::now();
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            return best;
        };
        double ti = bench(yi, ri), tn = bench(yn, rn);

        // C++ -O3 streaming reference (window-1 state per channel, emits y each step).
        std::vector<float> cy((size_t)D * T);
        double tc = 0;
        {
            const float *ap = a.data(), *bp = b.data(), *cp = c.data(), *xp = x.data();
            // a(d,n,t)=ap[d+D*(n+N*t)], b(n,t)=bp[n+N*t], x(d,t)=xp[d+D*t].
            auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel
            {
                std::vector<float> hs(N);
#pragma omp for schedule(static)
                for (int d = 0; d < D; d++) {
                    for (int nn = 0; nn < N; nn++)
                        hs[nn] = bp[nn] * xp[d];
                    float y0 = 0;
                    for (int nn = 0; nn < N; nn++)
                        y0 += cp[nn] * hs[nn];
                    cy[(size_t)d * T] = y0;
                    for (int t = 1; t < T; t++) {
                        float xdt = xp[(size_t)d + (size_t)t * D];
                        float yy = 0;
                        for (int nn = 0; nn < N; nn++) {
                            float av = ap[(size_t)d + (size_t)D * (nn + (size_t)N * t)];
                            float bv = bp[nn + (size_t)N * t];
                            hs[nn] = av * hs[nn] + bv * xdt;
                            yy += cp[nn + (size_t)N * t] * hs[nn];
                        }
                        cy[(size_t)d * T + t] = yy;
                    }
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            tc = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        double err = 0;
        bool bad = false;
        for (int d = 0; d < D; d++)
            for (int t = 0; t < T; t++) {
                float ai = ri(d, t), cn = rn(d, t), g = cy[(size_t)d * T + t];
                if (std::isnan(ai) || std::isnan(cn)) bad = true;
                err = std::max({err, (double)std::abs(ai - g), (double)std::abs(cn - g)});
            }
        double traj = (double)N * D * T * 4 / (1024.0 * 1024.0);
        printf("Mamba selective SSM  N=%d D=%d T=%d  (h trajectory %.0f MB)\n", N, D, T, traj);
        printf("  inductive Halide (fold h->2):      %8.3f ms\n", ti);
        printf("  non-inductive Halide (per-chan):   %8.3f ms\n", tn);
        printf("  (C++ sanity ref, ignore timing):   %8.3f ms\n", tc);
        printf("  max abs err %.3g%s -> %s\n", err, bad ? " (NaN)" : "",
               (!bad && err < 1e-2) ? "PASS" : "FAIL");

        // Dump inputs + inductive y for the fair Numba third-party bench.
        auto dump = [](const char *p, const float *dd, size_t nn) {
            FILE *f = fopen(p, "wb"); if (f) { fwrite(dd, sizeof(float), nn, f); fclose(f); } };
        dump("apps/mamba/a.bin", a.data(), (size_t)D * N * T);
        dump("apps/mamba/b.bin", b.data(), (size_t)N * T);
        dump("apps/mamba/c.bin", c.data(), (size_t)N * T);
        dump("apps/mamba/x.bin", x.data(), (size_t)D * T);
        std::vector<float> yflat((size_t)D * T);
        for (int d = 0; d < D; d++)
            for (int t = 0; t < T; t++)
                yflat[(size_t)d * T + t] = ri(d, t);
        dump("apps/mamba/y.bin", yflat.data(), (size_t)D * T);
        FILE *fp = fopen("apps/mamba/params.txt", "w");
        if (fp) {
            fprintf(fp, "%d %d %d %.6f\n", N, D, T, ti);
            fclose(fp);
        }
        return (!bad && err < 1e-2) ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
