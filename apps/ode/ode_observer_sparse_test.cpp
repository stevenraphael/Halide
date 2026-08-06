// SPARSE variant of the Allen-Cahn free-energy observer benchmark.
//
// Same physics dy/dt = eps*A*y + y - y^3, but the diffusion A is applied as the
// tridiagonal 3-point Laplacian stencil  (A y)_i = y_{i-1} - 2 y_i + y_{i+1}  in
// O(D) per step, instead of a dense O(D^2) matvec. Now per-step COMPUTE is O(D),
// the same order as the O(D) per-step state/observation traffic -- so once the
// materialized trajectory (O(D*B*T)) spills out of cache, its write/read bandwidth
// dominates and the folded (O(D*B*3)) inductive version should win on SPEED, not just
// memory. Observer = Ginzburg-Landau free energy (dissipating Lyapunov functional).
//
// Build (USE -O3):
//   g++ apps/ode/ode_observer_sparse_test.cpp -O3 -march=native -Idistrib/include \
//       -Lbuild/src -lHalide -lpthread -ldl -o /tmp/ode_sp -std=c++17
//   LD_LIBRARY_PATH=build/src HL_NUM_THREADS=1 /tmp/ode_sp [D B T]

#include "Halide.h"
#include <boost/numeric/odeint.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int D = argc > 1 ? atoi(argv[1]) : 1024;
    int B = argc > 2 ? atoi(argv[2]) : 1;
    int T = argc > 3 ? atoi(argv[3]) : 8192;
    const float h = 0.02f, eps = 0.1f;

    Buffer<float> y0(D, B);
    srand(5);
    for (int bb = 0; bb < B; bb++)
        for (int i = 0; i < D; i++)
            y0(i, bb) = (float)(rand() % 200) / 100.0f - 1.0f;

    auto bench = [&](int iters, auto fn) {
        double best = 1e18;
        for (int i = 0; i < iters; i++) {
            auto t0 = std::chrono::steady_clock::now();
            fn();
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms > 0) best = std::min(best, ms);
        }
        return best;
    };

    Var d("d"), b("b"), n("n");
    auto energy_density = [&](Func y, RVar rd_, Expr time) {
        Expr yi = y(rd_, b, time);
        Expr yip = y(clamp(cast<int>(rd_) + 1, 0, D - 1), b, time);
        Expr g = yip - yi;
        return 0.5f * eps * g * g + 0.25f * (1 - yi * yi) * (1 - yi * yi);
    };

    try {
        // ---------- 1. Inductive dynamics (stencil) + observer, state folded ----------
        Func y_i(Float(32), "y_i"), E_i("E_i");
        {
            y_i(d, b, n) = cast<float>(0);
            Expr nm1 = n - 1, nm2 = n - 2;
            // r.x = output component i, r.y = 3-point stencil offset {0,1,2}->{-1,0,+1}.
            // Both are RVars, so dim 0 never holds the output pure-var in a shifted
            // position -- it is a reduction dim, only n recurses. Work is 3*D = O(D).
            RDom r(0, D, 0, 3, "r");
            Expr i = r.x, off = r.y;
            Expr nb = clamp(i + off - 1, 0, D - 1);
            Expr w = select(off == 1, -2.0f, 1.0f);  // Laplacian stencil weights
            // Pointwise part (carry + AB2 reaction g(y)=y-y^3), gated to off==0 (once/i).
            Expr c1 = y_i(i, b, nm1), c2 = y_i(i, b, nm2);
            Expr react = 1.5f * (c1 - c1 * c1 * c1) - 0.5f * (c2 - c2 * c2 * c2);
            Expr onceper = c1 + h * react;
            // Diffusion eps*A*y accumulated over the 3-point stencil (AB2-weighted).
            Expr diff = h * eps * w * (1.5f * y_i(nb, b, nm1) - 0.5f * y_i(nb, b, nm2));
            // Seed at off==0 by ASSIGNMENT (not accumulate-onto-zero) so the folded slice
            // needs no per-step re-init; off>0 accumulate. off is the outer (unrolled) loop.
            y_i(i, b, n) = select(n <= 0, y0(i, b),
                                  select(off == 0, onceper + diff, y_i(i, b, n) + diff));

            RDom rd(0, D, "rd");
            E_i(b, n) = cast<float>(0);
            E_i(b, n) += energy_density(y_i, rd.x, n);
            E_i.compute_root();
            E_i.update(0).reorder(rd.x, b, n);
            y_i.compute_at(E_i, n).store_root().fold_storage(n, 3);
            y_i.update(0).allow_race_conditions().unroll(r.y).vectorize(r.x, 16);
            E_i.bound(b, 0, B).bound(n, 0, T);
            E_i.compile_jit();
        }

        // ---------- 2. Non-inductive: MATERIALIZE full trajectory, then reduce ----------
        Func y_m(Float(32), "y_m"), E_m("E_m");
        {
            RDom rn(1, T - 1, "rn");
            y_m(d, b, n) = undef<float>();
            y_m(d, b, 0) = y0(d, b);
            Expr p1 = rn - 1, p2 = clamp(rn - 2, 0, T - 1);
            auto lap = [&](Expr t) {
                return y_m(clamp(d - 1, 0, D - 1), b, t) - 2.0f * y_m(d, b, t) +
                       y_m(clamp(d + 1, 0, D - 1), b, t);
            };
            auto f = [&](Expr t) { Expr c = y_m(d, b, t); return eps * lap(t) + c - c * c * c; };
            y_m(d, b, rn) = y_m(d, b, p1) + h * (1.5f * f(p1) - 0.5f * f(p2));

            RDom rd(0, D, "rd");
            E_m(b, n) = cast<float>(0);
            E_m(b, n) += energy_density(y_m, rd.x, n);
            y_m.compute_root();  // full D*B*T
            y_m.update(0).unscheduled();
            y_m.update(1).reorder(d, b, rn).vectorize(d, 16);
            E_m.compute_root();
            E_m.update(0).reorder(rd.x, b, n);
            E_m.bound(b, 0, B).bound(n, 0, T);
            E_m.compile_jit();
        }

        Buffer<float> ei(B, T), em(B, T);
        E_i.realize(ei);
        E_m.realize(em);
        double t_i = bench(5, [&]() { E_i.realize(ei); });
        double t_m = bench(5, [&]() { E_m.realize(em); });

        auto energy = [&](const std::vector<float> &y) {
            double e = 0;
            for (int i = 0; i < D; i++) {
                float g = (i + 1 < D) ? (y[i + 1] - y[i]) : 0.f;
                e += 0.5 * eps * g * g + 0.25 * (1 - (double)y[i] * y[i]) * (1 - (double)y[i] * y[i]);
            }
            return (float)e;
        };
        auto rhs = [&](const std::vector<float> &y, std::vector<float> &f) {
            for (int i = 0; i < D; i++) {
                float lm = y[i > 0 ? i - 1 : 0], lp = y[i + 1 < D ? i + 1 : D - 1];
                float lap = lm - 2.f * y[i] + lp;
                f[i] = eps * lap + y[i] - y[i] * y[i] * y[i];
            }
        };

        // ---------- 3. Boost.odeint + observer ----------
        namespace odeint = boost::numeric::odeint;
        using state = std::vector<float>;
        auto sys = [&](const state &y, state &dy, double) { rhs(y, dy); };
        std::vector<float> eb((size_t)B * T);
        double t_boost = bench(5, [&]() {
            for (int bb = 0; bb < B; bb++) {
                state x(D);
                for (int i = 0; i < D; i++)
                    x[i] = y0(i, bb);
                int step = 0;
                eb[(size_t)bb * T + step++] = energy(x);
                odeint::adams_bashforth<2, state> ab;
                double t = 0, dt = h;
                ab.initialize(odeint::euler<state>(), sys, x, t, dt);
                eb[(size_t)bb * T + step++] = energy(x);
                for (int nn = 2; nn < T; nn++) {
                    ab.do_step(sys, x, t, dt);
                    t += dt;
                    eb[(size_t)bb * T + step++] = energy(x);
                }
            }
        });

        // ---------- 4. C++ reference + observer ----------
        std::vector<float> ec((size_t)B * T);
        double t_cpp = bench(5, [&]() {
            std::vector<float> yv(D), f1(D), f2(D), tmp(D);
            for (int bb = 0; bb < B; bb++) {
                for (int i = 0; i < D; i++)
                    yv[i] = y0(i, bb);
                rhs(yv, f1);
                f2 = f1;
                ec[(size_t)bb * T + 0] = energy(yv);
                for (int nn = 1; nn < T; nn++) {
                    for (int i = 0; i < D; i++)
                        tmp[i] = yv[i] + h * (1.5f * f1[i] - 0.5f * f2[i]);
                    yv = tmp;
                    f2 = f1;
                    rhs(yv, f1);
                    ec[(size_t)bb * T + nn] = energy(yv);
                }
            }
        });

        auto relerr = [&](auto get) {
            double e = 0;
            for (int bb = 0; bb < B; bb++)
                for (int nn = 0; nn < T; nn++) {
                    float ref = ec[(size_t)bb * T + nn];
                    e = std::max(e, (double)std::abs(get(bb, nn) - ref) / (std::abs(ref) + 1e-6f));
                }
            return e;
        };
        double err_i = relerr([&](int bb, int nn) { return ei(bb, nn); });
        double err_m = relerr([&](int bb, int nn) { return em(bb, nn); });
        double err_b = relerr([&](int bb, int nn) { return eb[(size_t)bb * T + nn]; });

        double fold_kb = (double)D * B * 3 * sizeof(float) / 1024.0;
        double full_kb = (double)D * B * T * sizeof(float) / 1024.0;
        printf("SPARSE Allen-Cahn free-energy observer  D=%d B=%d T=%d, single core\n", D, B, T);
        printf("  Halide inductive + observer:  %8.3f ms  (rel err %.2g)  state %.1f KB (D*B*3)\n", t_i, err_i, fold_kb);
        printf("  Halide non-ind (materialize): %8.3f ms  (rel err %.2g)  state %.1f KB (D*B*T)\n", t_m, err_m, full_kb);
        printf("  Boost.odeint + observer:      %8.3f ms  (rel err %.2g)\n", t_boost, err_b);
        printf("  C++ reference + observer:     %8.3f ms\n", t_cpp);
        printf("  speedup inductive vs materialize: %.2fx\n", t_m / t_i);
        bool pass = err_i < 1e-3 && err_m < 1e-3 && err_b < 1e-3;
        printf("  -> %s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    } catch (const Halide::Error &ex) {
        printf("HALIDE ERROR: %s\n", ex.what());
        return 2;
    }
}
