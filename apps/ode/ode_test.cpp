// Allen-Cahn reaction-diffusion ODE system dy/dt = eps*A*y + y - y^3 (A = discrete
// Laplacian), a NONLINEAR method-of-lines problem for which no matrix exponential
// exists, integrated with 2-step Adams-Bashforth (AB2), solved four ways and checked:
//
//   1. Halide INDUCTIVE     -- y is one inductive Func in time n; the matvec couples
//                              component i to every component k, so i and k are BOTH
//                              reduction variables of a 2-D RDom nested inside the
//                              inductive time n. fold_storage(n, 3) keeps 3 live time
//                              slices automatically.
//   2. Halide NON-INDUCTIVE -- the same recurrence as an ordinary RDom scan, with the
//                              time dimension folded BY HAND into a 3-slice ring
//                              (index n % 3). O(1)-in-time storage, like (1).
//   3. Boost.odeint         -- third-party baseline: adams_bashforth<2>, one Euler
//                              startup step (to match the app's f_{-1}=f_0 bootstrap),
//                              run per trajectory.
//   4. C++ reference        -- the same AB2 by hand; the correctness oracle.
//
// AB2:  y_n = y_{n-1} + h(1.5 f_{n-1} - 0.5 f_{n-2}),  f_m = eps*A*y_m + y_m - y_m^3.
// Bootstrap: at n=1, f_{-1} is clamped to f_0, i.e. y_1 = y_0 + h f_0 (forward Euler).
// All four output the FINAL state y_{T-1} (shape D x B), which is exactly what a
// third-party IVP integrator produces.
//
// Default configuration is a SINGLE trajectory (B=1) on a SINGLE core: a strictly
// per-ODE algorithmic comparison against Boost, with no batching or threading effects.
//
// Build (USE -O3: Boost.odeint and the C++ reference are template/loop code that only
// reaches full speed after inlining -- an unoptimized build makes them look ~10x slower
// than they are, while Halide's JIT'd pipeline is unaffected):
//   g++ apps/ode/ode_test.cpp -O3 -march=native -Iinclude -Lbuild/src -lHalide \
//       -lpthread -ldl -o /tmp/ode -std=c++17 ; LD_LIBRARY_PATH=build/src /tmp/ode [D B T]
// (Boost.odeint is header-only; no extra link flags.)

#include "Halide.h"
#include <boost/numeric/odeint.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    int D = argc > 1 ? atoi(argv[1]) : 512;  // system dimension
    int B = argc > 2 ? atoi(argv[2]) : 1;    // independent trajectories (single ODE)
    int T = argc > 3 ? atoi(argv[3]) : 512;  // timesteps
    const float h = 0.05f;
    const float eps = 0.1f;  // diffusion coefficient
    // Single core: neither Halide schedule uses parallel(), so both run serially; this
    // is a strictly per-ODE comparison (Boost and the C++ reference are serial too).

    // Allen-Cahn reaction-diffusion (method of lines): dy/dt = eps*A*y + y - y^3.
    // A = discrete Laplacian (tridiagonal, symmetric) is the diffusion operator; the
    // cubic reaction y - y^3 is the NONLINEAR term (bistable, solutions saturate to +/-1),
    // so there is no matrix exponential -- stepping is genuinely required. y0 random.
    Buffer<float> A(D, D), y0(D, B);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            A(j, i) = (i == j) ? -2.0f : ((i == j + 1 || j == i + 1) ? 1.0f : 0.0f);
    srand(5);
    for (int bb = 0; bb < B; bb++)
        for (int i = 0; i < D; i++) y0(i, bb) = (float)(rand() % 200) / 100.0f - 1.0f;

    // steady_clock + discard non-positive deltas: WSL2's high_resolution_clock can run
    // backwards across cores, producing spurious negative timings.
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

    // Column-major (D x B) final-state accessor: [i, bb] -> i*B + bb.
    auto at = [&](std::vector<float> &v, int i, int bb) -> float & { return v[(size_t)i * B + bb]; };
    auto maxdiff = [&](std::vector<float> &p, std::vector<float> &q) {
        double e = 0;
        bool bad = false;
        for (size_t i = 0; i < p.size(); i++) {
            if (std::isnan(p[i]) || std::isinf(p[i])) bad = true;
            e = std::max(e, (double)std::abs(p[i] - q[i]));
        }
        return bad ? 1e30 : e;
    };

    try {
        Var d("d"), b("b"), n("n");

        // ---------- 1. Halide INDUCTIVE (time folded via fold_storage) ----------
        Func y_ind(Float(32), "y_ind"), out_ind("out_ind");
        {
            // Pure (non-inductive) def: the per-time-slice accumulator starts at 0.
            y_ind(d, b, n) = cast<float>(0);

            // r.x = i (output component), r.y = k (matvec index). Both are RVars,
            // nested inside the one inductive dimension n (bounded lag 2).
            RDom r(0, D, 0, D, "r");
            Expr nm1 = max(0, n - 1), nm2 = max(0, n - 2);
            // Pointwise part (per output component r.x): the carry y_{n-1} plus the AB2
            // combination of the nonlinear reaction g(y) = y - y^3 at the two lags.
            Expr yq1 = y_ind(r.x, b, nm1), yq2 = y_ind(r.x, b, nm2);
            Expr react = 1.5f * (yq1 - yq1 * yq1 * yq1) - 0.5f * (yq2 - yq2 * yq2 * yq2);
            Expr onceper = yq1 + h * react;  // once-per-slice, gated to r.y == 0
            // Diffusion eps*A*y accumulated over the matvec index r.y (AB2-weighted).
            Expr diffusion = h * eps * A(r.y, r.x) *
                             (1.5f * y_ind(r.y, b, nm1) - 0.5f * y_ind(r.y, b, nm2));
            // One update. Monotone-in-n self-refs only; dim 0 is a reduction dim (r.y
            // there is fine). Base case n<=0 is y0. The pointwise term (carry + reaction)
            // is gated to r.y==0; the diffusion matvec accumulates over r.y.
            y_ind(r.x, b, n) = select(n <= 0, y0(r.x, b),
                                      y_ind(r.x, b, n) + cast<float>(r.y == 0) * onceper + diffusion);

            // Endpoint extract: keep only the final slice y_{T-1}. Time rn is OUTERMOST
            // (the matvec couples all components within a slice), which lets y fold.
            RDom rn(0, T, "rn");
            out_ind(d, b) = cast<float>(0);
            out_ind(d, b) += select(rn == T - 1, y_ind(d, b, rn), cast<float>(0));
            out_ind.update(0).reorder(d, b, rn);
            y_ind.compute_at(out_ind, rn).store_root().fold_storage(n, 3);
            y_ind.update(0).allow_race_conditions().vectorize(r.x, 4);
            out_ind.bound(d, 0, D).bound(b, 0, B);
            out_ind.compile_jit();
        }

        // ---------- 2. Halide NON-INDUCTIVE (manual mod-3 time ring) ----------
        Func y_ni(Float(32), "y_ni"), out_ni("out_ni");
        {
            Var s("s");
            RDom r(0, D, 1, T - 1);
            Expr rk = r.x, rn = r.y;
            Expr cur = rn % 3;                       // slot for time n
            Expr prev1 = (rn - 1) % 3;               // slot for n-1 (rn>=1 => nonneg)
            Expr prev2 = clamp(rn - 2, 0, T - 1) % 3;  // n=1 -> slot 0 (=y0): Euler boot
            y_ni(d, b, s) = undef<float>();
            y_ni(d, b, 0) = y0(d, b);                // n=0 lives in slot 0 (init, update 0)
            // At rk==0 seed with the pointwise part (carry y_{n-1} + AB2 reaction), a
            // plain assignment since the reused ring slot holds stale data; accumulate
            // the diffusion matvec for rk>0. Same arithmetic as the inductive version.
            Expr yq1 = y_ni(d, b, prev1), yq2 = y_ni(d, b, prev2);
            Expr react = 1.5f * (yq1 - yq1 * yq1 * yq1) - 0.5f * (yq2 - yq2 * yq2 * yq2);
            Expr seed = select(rk == 0, yq1 + h * react, y_ni(d, b, cur));
            Expr diffusion = h * eps * A(rk, d) * (1.5f * y_ni(rk, b, prev1) - 0.5f * y_ni(rk, b, prev2));
            y_ni(d, b, cur) = seed + diffusion;

            out_ni(d, b) = y_ni(d, b, (T - 1) % 3);  // final state from slot (T-1)%3
            y_ni.compute_root();
            y_ni.update(0).unscheduled();  // slot-0 init needs no schedule
            // Vectorize the pure component dim d (safe: distinct d are independent), with
            // the matvec index r.x as the reduction loop -- level SIMD footing with the
            // inductive schedule.
            y_ni.update(1).reorder(d, r.x, r.y, b).vectorize(d, 4);
            out_ni.bound(d, 0, D).bound(b, 0, B);
            out_ni.compile_jit();
        }

        Buffer<float> hb_ind(D, B), hb_ni(D, B);
        out_ind.realize(hb_ind);
        out_ni.realize(hb_ni);
        double t_ind = bench(5, [&]() { out_ind.realize(hb_ind); });
        double t_ni = bench(5, [&]() { out_ni.realize(hb_ni); });

        // ---------- 3. Boost.odeint adams_bashforth<2> (third-party baseline) ----------
        using state_type = std::vector<float>;
        namespace odeint = boost::numeric::odeint;
        auto sys = [&](const state_type &yv, state_type &dydt, double /*t*/) {
            for (int i = 0; i < D; i++) {
                float s = 0;
                for (int k = 0; k < D; k++) s += A(k, i) * yv[k];        // diffusion (A y)_i
                dydt[i] = eps * s + yv[i] - yv[i] * yv[i] * yv[i];       // + reaction y - y^3
            }
        };
        std::vector<float> boost_final((size_t)D * B);
        double t_boost = bench(5, [&]() {
            for (int bb = 0; bb < B; bb++) {
                state_type x(D);
                for (int i = 0; i < D; i++) x[i] = y0(i, bb);
                odeint::adams_bashforth<2, state_type> ab;
                double t = 0, dt = h;
                // One Euler startup step -> x = x_1, records f_0 (matches app bootstrap).
                ab.initialize(odeint::euler<state_type>(), sys, x, t, dt);
                for (int nn = 2; nn < T; nn++) {
                    ab.do_step(sys, x, t, dt);
                    t += dt;
                }
                for (int i = 0; i < D; i++) boost_final[(size_t)i * B + bb] = x[i];
            }
        });

        // ---------- 4. C++ reference AB2 (correctness oracle) ----------
        std::vector<float> cref((size_t)D * B);
        double t_cpp = bench(5, [&]() {
            std::vector<float> yv(D), f1(D), f2(D), tmp(D);
            for (int bb = 0; bb < B; bb++) {
                for (int i = 0; i < D; i++) yv[i] = y0(i, bb);
                auto rhs = [&](std::vector<float> &v, std::vector<float> &f) {
                    for (int i = 0; i < D; i++) {
                        float s = 0;
                        for (int k = 0; k < D; k++) s += A(k, i) * v[k];
                        f[i] = eps * s + v[i] - v[i] * v[i] * v[i];  // eps*A*y + y - y^3
                    }
                };
                rhs(yv, f1);
                f2 = f1;  // f_{-1} := f_0 bootstrap
                for (int nn = 1; nn < T; nn++) {
                    for (int i = 0; i < D; i++) tmp[i] = yv[i] + h * (1.5f * f1[i] - 0.5f * f2[i]);
                    for (int i = 0; i < D; i++) yv[i] = tmp[i];
                    for (int i = 0; i < D; i++) f2[i] = f1[i];
                    rhs(yv, f1);
                }
                for (int i = 0; i < D; i++) at(cref, i, bb) = yv[i];
            }
        });

        // ---------- compare final states ----------
        std::vector<float> vind((size_t)D * B), vni((size_t)D * B);
        for (int i = 0; i < D; i++)
            for (int bb = 0; bb < B; bb++) {
                at(vind, i, bb) = hb_ind(i, bb);
                at(vni, i, bb) = hb_ni(i, bb);
            }
        double e_ind = maxdiff(vind, cref);
        double e_ni = maxdiff(vni, cref);
        double e_boost = maxdiff(boost_final, cref);

        printf("Allen-Cahn (dy/dt=eps*A*y+y-y^3, AB2), final state  D=%d B=%d T=%d\n", D, B, T);
        printf("  Halide inductive (fold_storage):  %8.3f ms  (err vs C++ %.3g)\n", t_ind, e_ind);
        printf("  Halide non-inductive (%%3 ring):   %8.3f ms  (err vs C++ %.3g)\n", t_ni, e_ni);
        printf("  Boost.odeint adams_bashforth<2>:  %8.3f ms  (err vs C++ %.3g)\n", t_boost, e_boost);
        printf("  C++ reference (same AB2):         %8.3f ms\n", t_cpp);

        bool pass = e_ind < 1e-2 && e_ni < 1e-2 && e_boost < 1e-2;
        printf("  -> %s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    } catch (const Halide::Error &ex) {
        printf("HALIDE ERROR: %s\n", ex.what());
        return 2;
    }
}
