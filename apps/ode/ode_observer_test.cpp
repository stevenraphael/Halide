// Per-step OBSERVER benchmark for the Allen-Cahn system dy/dt = eps*A*y + y - y^3.
//
// The observation is the Ginzburg-Landau FREE ENERGY, the Lyapunov functional the
// dynamics dissipate (so it must decrease monotonically -- a real integrator-validation
// diagnostic):
//     E(y) = sum_i [ (eps/2)(y_{i+1} - y_i)^2 + (1/4)(1 - y_i^2)^2 ]
// It is a reduction over the WHOLE state slice, one scalar per trajectory per step.
// Output is the energy trajectory, shape (B, T) -- O(B*T), independent of D.
//
// Four implementations, all AB2 with the one-Euler bootstrap, cross-checked:
//   1. Halide INDUCTIVE  -- observer is an ordinary Func at the folded time loop; the
//                           D-dim state y is kept to 3 slices (fold_storage). O(1) state.
//   2. Halide NON-IND.   -- to feed a per-step observer the %3 ring has no consumer
//                           boundary, so it must MATERIALIZE the full y(D,B,T), then
//                           reduce. O(T) state.
//   3. Boost.odeint      -- observer computes E(x) each step in the do_step loop.
//   4. C++ reference     -- correctness oracle.
//
// Build (USE -O3):
//   g++ apps/ode/ode_observer_test.cpp -O3 -march=native -Idistrib/include -Lbuild/src \
//       -lHalide -lpthread -ldl -o /tmp/ode_obs -std=c++17
//   LD_LIBRARY_PATH=build/src HL_NUM_THREADS=1 /tmp/ode_obs [D B T]

#include "Halide.h"
#include <boost/numeric/odeint.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int D = argc > 1 ? atoi(argv[1]) : 512;
    int B = argc > 2 ? atoi(argv[2]) : 1;
    int T = argc > 3 ? atoi(argv[3]) : 512;
    const float h = 0.05f, eps = 0.1f;

    Buffer<float> A(D, D), y0(D, B);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            A(j, i) = (i == j) ? -2.0f : ((i == j + 1 || j == i + 1) ? 1.0f : 0.0f);
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

    // Energy density Func factory: given a state Func y(comp, b, time), build the AB2
    // dynamics update, then reduce each slice to the free energy obs(b, time).
    auto make_energy = [&](Func y, RDom rd) {
        Expr yi = y(rd, b, n);
        Expr yip = y(clamp(rd + 1, 0, D - 1), b, n);  // neighbor (clamped at boundary)
        Expr grad = yip - yi;                         // 0 at the last node
        return 0.5f * eps * grad * grad + 0.25f * (1 - yi * yi) * (1 - yi * yi);
    };

    try {
        // ---------- 1. Inductive dynamics + observer, state folded ----------
        Func y_i(Float(32), "y_i"), E_i("E_i");
        {
            y_i(d, b, n) = cast<float>(0);
            RDom r(0, D, 0, D, "r");
            Expr nm1 = max(0, n - 1), nm2 = max(0, n - 2);
            Expr yq1 = y_i(r.x, b, nm1), yq2 = y_i(r.x, b, nm2);
            Expr react = 1.5f * (yq1 - yq1 * yq1 * yq1) - 0.5f * (yq2 - yq2 * yq2 * yq2);
            Expr onceper = yq1 + h * react;
            Expr diff = h * eps * A(r.y, r.x) * (1.5f * y_i(r.y, b, nm1) - 0.5f * y_i(r.y, b, nm2));
            y_i(r.x, b, n) = select(n <= 0, y0(r.x, b), y_i(r.x, b, n) + cast<float>(r.y == 0) * onceper + diff);

            RDom rd(0, D, "rd");
            E_i(b, n) = cast<float>(0);
            E_i(b, n) += make_energy(y_i, rd);
            E_i.compute_root();
            E_i.update(0).reorder(rd, b, n);
            y_i.compute_at(E_i, n).store_root().fold_storage(n, 3);
            y_i.update(0).allow_race_conditions().vectorize(r.x, 4);
            E_i.bound(b, 0, B).bound(n, 0, T);
            E_i.compile_jit();
        }

        // ---------- 2. Non-inductive: MATERIALIZE full trajectory, then reduce ----------
        Func y_m(Float(32), "y_m"), E_m("E_m");
        {
            RDom r(0, D, 1, T - 1);
            Expr rk = r.x, rn = r.y;
            y_m(d, b, n) = undef<float>();
            y_m(d, b, 0) = y0(d, b);
            Expr yq1 = y_m(d, b, rn - 1), yq2 = y_m(d, b, clamp(rn - 2, 0, T - 1));
            Expr react = 1.5f * (yq1 - yq1 * yq1 * yq1) - 0.5f * (yq2 - yq2 * yq2 * yq2);
            Expr seed = select(rk == 0, yq1 + h * react, y_m(d, b, rn));
            Expr diff = h * eps * A(rk, d) * (1.5f * y_m(rk, b, rn - 1) - 0.5f * y_m(rk, b, clamp(rn - 2, 0, T - 1)));
            y_m(d, b, rn) = seed + diff;

            RDom rd(0, D, "rd");
            E_m(b, n) = cast<float>(0);
            E_m(b, n) += make_energy(y_m, rd);
            y_m.compute_root();  // full D*B*T materialized
            y_m.update(0).unscheduled();
            y_m.update(1).reorder(d, r.x, r.y, b).vectorize(d, 4);
            E_m.compute_root();
            E_m.update(0).reorder(rd, b, n);
            E_m.bound(b, 0, B).bound(n, 0, T);
            E_m.compile_jit();
        }

        Buffer<float> ei(B, T), em(B, T);
        E_i.realize(ei);
        E_m.realize(em);
        double t_i = bench(5, [&]() { E_i.realize(ei); });
        double t_m = bench(5, [&]() { E_m.realize(em); });

        // ---------- energy helper for the imperative baselines ----------
        auto energy = [&](const std::vector<float> &y) {
            double e = 0;
            for (int i = 0; i < D; i++) {
                float g = (i + 1 < D) ? (y[i + 1] - y[i]) : 0.f;
                e += 0.5 * eps * g * g + 0.25 * (1 - (double)y[i] * y[i]) * (1 - (double)y[i] * y[i]);
            }
            return (float)e;
        };

        // ---------- 3. Boost.odeint + observer ----------
        namespace odeint = boost::numeric::odeint;
        using state = std::vector<float>;
        auto sys = [&](const state &y, state &dy, double) {
            for (int i = 0; i < D; i++) {
                float s = 0;
                for (int k = 0; k < D; k++)
                    s += A(k, i) * y[k];
                dy[i] = eps * s + y[i] - y[i] * y[i] * y[i];
            }
        };
        std::vector<float> eb((size_t)B * T);
        double t_boost = bench(5, [&]() {
            for (int bb = 0; bb < B; bb++) {
                state x(D);
                for (int i = 0; i < D; i++)
                    x[i] = y0(i, bb);
                int step = 0;
                eb[(size_t)bb * T + step++] = energy(x);  // n=0
                odeint::adams_bashforth<2, state> ab;
                double t = 0, dt = h;
                ab.initialize(odeint::euler<state>(), sys, x, t, dt);
                eb[(size_t)bb * T + step++] = energy(x);  // n=1
                for (int nn = 2; nn < T; nn++) {
                    ab.do_step(sys, x, t, dt);
                    t += dt;
                    eb[(size_t)bb * T + step++] = energy(x);  // n
                }
            }
        });

        // ---------- 4. C++ reference + observer ----------
        std::vector<float> ec((size_t)B * T);
        double t_cpp = bench(5, [&]() {
            std::vector<float> yv(D), f1(D), f2(D), tmp(D);
            auto rhs = [&](std::vector<float> &v, std::vector<float> &f) {
                for (int i = 0; i < D; i++) {
                    float s = 0;
                    for (int k = 0; k < D; k++)
                        s += A(k, i) * v[k];
                    f[i] = eps * s + v[i] - v[i] * v[i] * v[i];
                }
            };
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

        // ---------- correctness (relative) + monotonicity check ----------
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
        bool mono = true;  // energy must dissipate (allow tiny float slack)
        for (int bb = 0; bb < B; bb++)
            for (int nn = 1; nn < T; nn++)
                if (ec[(size_t)bb * T + nn] > ec[(size_t)bb * T + nn - 1] + 1e-3f) mono = false;

        double fold_kb = (double)D * B * 3 * sizeof(float) / 1024.0;
        double full_kb = (double)D * B * T * sizeof(float) / 1024.0;
        printf("Allen-Cahn free-energy observer  D=%d B=%d T=%d, single core\n", D, B, T);
        printf("  Halide inductive + observer:  %8.3f ms  (rel err %.2g)  state %.1f KB (D*B*3)\n", t_i, err_i, fold_kb);
        printf("  Halide non-ind (materialize): %8.3f ms  (rel err %.2g)  state %.1f KB (D*B*T)\n", t_m, err_m, full_kb);
        printf("  Boost.odeint + observer:      %8.3f ms  (rel err %.2g)  state ~%.1f KB (O(D))\n", t_boost, err_b, (double)D * sizeof(float) / 1024.0);
        printf("  C++ reference + observer:     %8.3f ms\n", t_cpp);
        printf("  energy dissipates (E_n <= E_{n-1}): %s   E[0]=%.4g  E[T-1]=%.4g\n",
               mono ? "yes" : "NO", ec[0], ec[T - 1]);
        bool pass = err_i < 1e-3 && err_m < 1e-3 && err_b < 1e-3 && mono;
        printf("  -> %s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    } catch (const Halide::Error &ex) {
        printf("HALIDE ERROR: %s\n", ex.what());
        return 2;
    }
}
