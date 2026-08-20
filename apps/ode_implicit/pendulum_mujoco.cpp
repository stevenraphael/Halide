// Fixed-step "implicit" damped-pendulum integration, EXACTLY reproducing
// MuJoCo's `integrator="implicit"` step for a single damped hinge pendulum
// (see pendulum_mujoco_check.py for the MuJoCo model + the derivation of
// the closed-form qfrc_bias(theta) = -2.4525*cos(theta) fit and the
// implicit-mass-augmentation formula, verified against MuJoCo to ~1e-16).
//
// This is the real, literal fixed-step benchmark the earlier discussion
// pointed to: MuJoCo (like Bullet/PhysX and dSPACE/OPAL-RT HIL platforms)
// uses a FIXED step size and, for its "implicit" integrator, folds joint
// damping into the effective mass matrix (a linear solve -- here 1x1/scalar
// since it's a single DOF, but the same augmented-mass structure used for
// multi-DOF systems) rather than adaptive step size or adaptive iteration
// count, precisely because real-time/robotics/HIL simulation needs
// deterministic per-step cost.
//
// MuJoCo's formula per step:
//   M_eff = M + h*damping                          (augmented mass)
//   qacc  = (qfrc_bias_neg(theta) - damping*omega) / M_eff
//         = (2.4525*cos(theta) - damping*omega) / M_eff
//   omega_new = omega + h*qacc
//   theta_new = theta + h*omega_new                 (semi-implicit position update)
// No Newton iteration needed here (damping enters linearly), so both the
// inductive and non-inductive Halide schedules just materialize/fold this
// single-shift recursion -- same "small per-step state, forced-fold
// argument" as ode_implicit.cpp, just with a genuinely real fixed-step
// mainstream physics engine as the third-party comparison instead of an
// ODE-solver library.
//
// Build: g++ apps/ode_implicit/pendulum_mujoco.cpp -O3 -march=native -fopenmp
//        -Iinclude -Lbuild/src -lHalide -lpthread -ldl -o /tmp/pendmj
//        -std=c++17 ; LD_LIBRARY_PATH=build/src /tmp/pendmj [B T]

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4096;
    int T = argc > 2 ? atoi(argv[2]) : 100000;

    const double h_d = 0.001;
    const double damping_d = 2.0;
    const double M_d = 0.08573594936708857;
    const double A_d = 2.4525;
    const double M_eff_d = M_d + h_d * damping_d;

    Expr h = Expr(h_d), damping = Expr(damping_d), A = Expr(A_d), inv_M_eff = Expr(1.0 / M_eff_d);

    Buffer<double> theta0(B), omega0(B);
    srand(3);
    for (int b = 0; b < B; b++) {
        theta0(b) = (rand() % 300) / 100.0 - 1.5;  // spread of initial angles, [-1.5, 1.5)
        omega0(b) = 0.0;
    }

    auto build = [&](bool inductive) -> Func {
        Var b("b"), t("t");
        Func state(std::vector<Type>{Float(64), Float(64)}, inductive ? "state_ind" : "state_mat");

        auto step = [&](Expr theta, Expr omega) -> std::pair<Expr, Expr> {
            Expr qacc = (A * cos(theta) - damping * omega) * inv_M_eff;
            Expr omega_new = omega + h * qacc;
            Expr theta_new = theta + h * omega_new;
            return {theta_new, omega_new};
        };

        if (inductive) {
            Expr prev_theta = state(b, t - 1)[0], prev_omega = state(b, t - 1)[1];
            auto next = step(prev_theta, prev_omega);
            state(b, t) = Tuple(
                select(t <= 0, theta0(b), likely(next.first)),
                select(t <= 0, omega0(b), likely(next.second)));
        } else {
            state(b, t) = Tuple(theta0(b), omega0(b));
            RDom rt(1, T - 1, "rt");
            Expr prev_theta = state(b, rt - 1)[0], prev_omega = state(b, rt - 1)[1];
            auto next = step(prev_theta, prev_omega);
            state(b, rt) = Tuple(next.first, next.second);
        }

        Func obs("obs");
        obs(b, t) = state(b, t)[0];  // pointwise (non-reduction): observe theta directly

        const int V = 8;
        Var bo("bo"), bi("bi");
        obs.bound(b, 0, B).bound(t, 0, T)
           .split(b, bo, bi, V).reorder(bi, t, bo).vectorize(bi).parallel(bo);
        if (inductive) {
            state.reorder_storage(b, t)
                 .compute_at(obs, t).store_at(obs, bo).fold_storage(t, 2).vectorize(b, V);
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
        Buffer<double> ri(B, T), rn(B, T);
        oi.realize(ri); on.realize(rn);

        auto bench = [&](Func f, Buffer<double> &bb) {
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

        double err_ind_ni = 0;
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++)
            err_ind_ni = std::max(err_ind_ni, std::abs(ri(b, t) - rn(b, t)));
        printf("  inductive vs non-inductive (Halide-vs-Halide) max diff: %g\n", err_ind_ni);

        double traj = (double)2 * B * T * 8 / (1024.0 * 1024.0);
        printf("MuJoCo-matching damped pendulum (implicit, fixed step h=%.4f)  B=%d T=%d  (trajectory %.1f MB)\n",
               h_d, B, T, traj);
        printf("  inductive Halide (fold state->2):  %8.3f ms\n", ti);
        printf("  non-inductive Halide (materialize): %8.3f ms\n", tn);

        auto dump = [](const char *p, const double *d, size_t n) {
            FILE *f = fopen(p, "wb"); if (f) { fwrite(d, sizeof(double), n, f); fclose(f); } };
        dump("apps/ode_implicit/pend_theta0.bin", theta0.data(), B);
        std::vector<double> yflat((size_t)B * T);
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) yflat[(size_t)b * T + t] = ri(b, t);
        dump("apps/ode_implicit/pend_obs.bin", yflat.data(), (size_t)B * T);
        FILE *fp = fopen("apps/ode_implicit/pend_params.txt", "w");
        if (fp) { fprintf(fp, "%d %d %.6f\n", B, T, ti); fclose(fp); }
        return 0;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
