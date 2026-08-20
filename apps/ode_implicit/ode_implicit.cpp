// Implicit ODE integrator benchmark: backward-Euler on the stiff Van der Pol
// oscillator, with a fixed number of Newton iterations (a small 2x2 linear
// solve, closed-form inverse) nested inside the per-timestep update, and a
// per-step consumer that is a plain POINTWISE function of the state (not a
// reduction/sum) -- so there is no reduction structure to hide the fold
// behind: a non-inductive schedule MUST materialize the whole (B x T x 2)
// trajectory, while inductive folds state down to O(1) per trajectory.
//
//   y1' = y2
//   y2' = mu*(1 - y1^2)*y2 - y1                  (Van der Pol, stiff for large mu)
//
// Problem parameters (mu, y0, t_final) are the CANONICAL stiff Van der Pol
// test problem from the classical ODE test sets (Hairer & Wanner "VDPOL";
// used verbatim as an example/benchmark problem across mainstream solvers --
// MATLAB's ode15s/ode23s documentation, SUNDIALS/CVODE demos,
// DifferentialEquations.jl's test suite): mu=1000, y0=(2,0), t in [0,2].
// Using the actual textbook/solver-example parameters (rather than inventing
// our own mu/initial condition) means "is this a realistic stiff problem" is
// not something we get to decide by picking convenient numbers.
//
// Backward Euler: y_{n+1} = y_n + h f(y_{n+1}), solved via K fixed Newton
// steps (closed-form 2x2 Jacobian inverse, unrolled in C++ -- same pattern
// as kalman_ss.cpp's manually-unrolled matvec, since Halide's inductive
// self-reference must sit lexically inside a select(), which rules out an
// RDom-driven Newton loop here too).
//
// Consumer: obs(b,t) = tanh(y1(b,t))  -- pointwise, NOT a sum, unlike
// kalman_ss's C.x dot product. This is the "forced fold" case: nothing
// downstream reduces the trajectory away, so if inductive didn't fold state
// storage, every intermediate Newton iterate for every timestep would need
// to be materialized.
//
// Build: g++ apps/ode_implicit/ode_implicit.cpp -O3 -march=native -Iinclude
//        -Lbuild/src -lHalide -lpthread -ldl -o /tmp/odei -std=c++17
//        LD_LIBRARY_PATH=build/src /tmp/odei [B T mu h K]

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    // Defaults are the canonical VDPOL test-problem parameters (mu, y0,
    // t_final) plus a step size chosen so the fixed-step trajectory actually
    // tracks the true solution (verified against scipy LSODA over the full
    // [0,2] test interval -- see ode_implicit_lsoda.py). At mu=1000 the
    // solution has a sharp relaxation transition around t~1.6, so h must be
    // small enough to resolve it even though backward Euler is unconditionally
    // stable there; h=1e-5 (T=200001 steps to t_final=2) does.
    int B = argc > 1 ? atoi(argv[1]) : 4096;      // batch of independent trajectories
    int T = argc > 2 ? atoi(argv[2]) : 200001;    // timesteps
    float mu = argc > 3 ? (float)atof(argv[3]) : 1000.0f;    // canonical VDPOL stiffness
    float h = argc > 4 ? (float)atof(argv[4]) : 1e-5f;       // step size
    int K = argc > 5 ? atoi(argv[5]) : 3;         // fixed Newton iteration count

    // Batch of trajectories clustered around the canonical VDPOL initial
    // condition y0=(2,0), perturbed slightly so the B trajectories are
    // distinct (not identical, so vectorized/parallel schedules do real work)
    // while still starting from essentially the textbook initial state.
    Buffer<float> y1_0(B), y2_0(B);
    srand(7);
    for (int b = 0; b < B; b++) {
        y1_0(b) = 2.0f + (rand() % 100) / 1000.0f - 0.05f;   // 2.0 +/- 0.05
        y2_0(b) = (rand() % 100) / 1000.0f - 0.05f;          // 0.0 +/- 0.05
    }

    // One closed-form Newton step: given current iterate (cy1,cy2) and the
    // previous-timestep state (a,b), returns the next iterate. Residual
    // F(y) = y - yn - h*f(y); Jacobian is 2x2, inverted in closed form.
    auto newton_step = [&](Expr cy1, Expr cy2, Expr a, Expr b) -> std::pair<Expr, Expr> {
        Expr f1 = cy2;
        Expr f2 = mu * (1.0f - cy1 * cy1) * cy2 - cy1;
        Expr F1 = cy1 - a - h * f1;
        Expr F2 = cy2 - b - h * f2;
        Expr J11 = 1.0f;
        Expr J12 = -h;
        Expr J21 = h * (2.0f * mu * cy1 * cy2 + 1.0f);
        Expr J22 = 1.0f - h * mu * (1.0f - cy1 * cy1);
        Expr det = J11 * J22 - J12 * J21;
        Expr inv_det = 1.0f / det;
        Expr delta1 = (J22 * F1 - J12 * F2) * inv_det;
        Expr delta2 = (-J21 * F1 + J11 * F2) * inv_det;
        return {cy1 - delta1, cy2 - delta2};
    };
    // K fixed Newton iterations, unrolled in C++, starting from the
    // previous state as the initial guess (standard for small h).
    auto solve = [&](Expr a, Expr b) -> std::pair<Expr, Expr> {
        Expr cy1 = a, cy2 = b;
        for (int k = 0; k < K; k++) {
            auto next = newton_step(cy1, cy2, a, b);
            cy1 = next.first;
            cy2 = next.second;
        }
        return {cy1, cy2};
    };

    auto build = [&](bool inductive) -> Func {
        Var b("b"), t("t");
        Func state(std::vector<Type>{Float(32), Float(32)}, inductive ? "state_ind" : "state_mat");

        if (inductive) {
            Expr prev1 = state(b, t - 1)[0];
            Expr prev2 = state(b, t - 1)[1];
            auto sol = solve(prev1, prev2);
            state(b, t) = Tuple(
                select(t <= 0, y1_0(b), likely(sol.first)),
                select(t <= 0, y2_0(b), likely(sol.second)));
        } else {
            // FAIR non-inductive: identical per-step Newton solve, but as an
            // RDom-over-time scan (materializes instead of folding).
            state(b, t) = Tuple(y1_0(b), y2_0(b));
            RDom rt(1, T - 1, "rt");
            Expr prev1 = state(b, rt - 1)[0];
            Expr prev2 = state(b, rt - 1)[1];
            auto sol = solve(prev1, prev2);
            state(b, rt) = Tuple(sol.first, sol.second);
        }

        // Pointwise (non-reduction) per-step consumer.
        Func obs("obs");
        obs(b, t) = tanh(state(b, t)[0]);

        const int V = 8;
        Var bo("bo"), bi("bi");
        obs.bound(b, 0, B).bound(t, 0, T)
           .split(b, bo, bi, V).reorder(bi, t, bo).vectorize(bi).parallel(bo);
        if (inductive) {
            state.reorder_storage(b, t)
                 .compute_at(obs, t).store_at(obs, bo).fold_storage(t, 1).vectorize(b, V);
        } else if (getenv("MATROOT")) {
            state.reorder_storage(b, t).compute_root().vectorize(b, V);
            state.update(0).vectorize(b, V);
        } else {
            state.reorder_storage(b, t).compute_at(obs, bo).vectorize(b, V);
            state.update(0).vectorize(b, V);
        }
        return obs;
    };

    try {
        Func oi = build(true), on = build(false);
        oi.compile_jit(); on.compile_jit();
        Buffer<float> ri(B, T), rn(B, T);
        oi.realize(ri); on.realize(rn);

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
        double ti = bench(oi, ri), tn = bench(on, rn);

        // C++ -O3 streaming reference (window-1 state, emits obs each step).
        std::vector<float> cy((size_t)B * T);
        double tc = 0;
        {
            const float *ya = y1_0.data(), *yb = y2_0.data();
            auto t0 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < B; b++) {
                float a = ya[b], bb2 = yb[b];
                cy[(size_t)b * T] = std::tanh(a);
                for (int t = 1; t < T; t++) {
                    float cy1 = a, cy2 = bb2;
                    for (int k = 0; k < K; k++) {
                        float f1 = cy2;
                        float f2 = mu * (1.0f - cy1 * cy1) * cy2 - cy1;
                        float F1 = cy1 - a - h * f1;
                        float F2 = cy2 - bb2 - h * f2;
                        float J11 = 1.0f, J12 = -h;
                        float J21 = h * (2.0f * mu * cy1 * cy2 + 1.0f);
                        float J22 = 1.0f - h * mu * (1.0f - cy1 * cy1);
                        float det = J11 * J22 - J12 * J21;
                        float inv_det = 1.0f / det;
                        float d1 = (J22 * F1 - J12 * F2) * inv_det;
                        float d2 = (-J21 * F1 + J11 * F2) * inv_det;
                        cy1 -= d1; cy2 -= d2;
                    }
                    a = cy1; bb2 = cy2;
                    cy[(size_t)b * T + t] = std::tanh(a);
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            tc = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        double err = 0; bool bad = false;
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) {
            float a = ri(b, t), c = rn(b, t), g = cy[(size_t)b * T + t];
            if (std::isnan(a) || std::isnan(c)) bad = true;
            err = std::max({err, (double)std::abs(a - g), (double)std::abs(c - g)});
        }
        double traj = (double)2 * B * T * 4 / (1024.0 * 1024.0);
        printf("Implicit ODE (backward-Euler Van der Pol, K=%d Newton steps)  B=%d T=%d mu=%.2f h=%.4f  (trajectory %.1f MB)\n",
               K, B, T, mu, h, traj);
        printf("  inductive Halide (fold state->2):  %8.3f ms\n", ti);
        printf("  non-inductive Halide (materialize): %8.3f ms\n", tn);
        printf("  (C++ sanity ref, ignore timing):    %8.3f ms\n", tc);
        printf("  max abs err %.3g%s -> %s\n", err, bad ? " (NaN)" : "", (!bad && err < 1e-2) ? "PASS" : "FAIL");

        // Dump initial conditions, params, and the inductive result for a
        // fair third-party (Numba/JAX) comparison on identical data.
        auto dump = [](const char *p, const float *d, size_t n) {
            FILE *f = fopen(p, "wb"); if (f) { fwrite(d, sizeof(float), n, f); fclose(f); } };
        dump("apps/ode_implicit/y1_0.bin", y1_0.data(), B);
        dump("apps/ode_implicit/y2_0.bin", y2_0.data(), B);
        std::vector<float> yflat((size_t)B * T);
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) yflat[(size_t)b * T + t] = ri(b, t);
        dump("apps/ode_implicit/obs.bin", yflat.data(), (size_t)B * T);
        FILE *fp = fopen("apps/ode_implicit/params.txt", "w");
        if (fp) { fprintf(fp, "%d %d %.6f %.6f %d %.6f\n", B, T, mu, h, K, ti); fclose(fp); }
        return (!bad && err < 1e-2) ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
