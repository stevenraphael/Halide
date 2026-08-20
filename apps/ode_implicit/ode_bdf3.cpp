// Fixed-step, fixed-Newton-iteration BDF3 (3rd-order Backward Differentiation
// Formula) on the canonical stiff Van der Pol test problem (mu=1000,
// y0=(2,0), t in [0,2] -- see ode_implicit.cpp for why these are the
// standard VDPOL parameters, not invented ones).
//
// DOUBLE PRECISION: needed to make an atol=1e-9 comparison against SUNDIALS
// CVODE (which is itself compiled double-precision) meaningful -- float32
// has ~1.2e-7 relative precision, so for O(1)-magnitude values like tanh(y1)
// it CANNOT resolve differences below ~1e-7 no matter how small h gets (we
// hit that floor directly: shrinking h from 1e-5 to 1e-6 only got the
// internal Halide-vs-C++-reference diff to 6e-8, the float32 noise floor,
// not the truncation-error floor). Float64 error floor is ~1e-16, so with a
// small enough h an atol=1e-9 target is actually achievable and meaningful.
//
// Unlike backward Euler (which only needs y_{n-1}), BDF3 needs a 3-step
// history:
//   (11/6) y_{n+1} - 3 y_n + (3/2) y_{n-1} - (1/3) y_{n-2} = h f(y_{n+1})
// solved via K fixed Newton iterations (same closed-form 2x2 Jacobian
// pattern as ode_implicit.cpp, generalized to an arbitrary leading
// coefficient `a` and right-hand-side `rhs`). Startup uses lower-order
// formulas for the first two steps (BDF1 then BDF2), since a 3-step history
// isn't available yet.
//
// This is the case that's meant to WIDEN inductive folding's advantage over
// ode_implicit.cpp's plain backward Euler: needing a window of the last 3
// states (not just 1) means the non-inductive/materializing schedule must
// still write out the WHOLE trajectory (same as before), but inductive only
// ever needs fold_storage(t, 4) -- current + 3 history slots -- so the
// memory-bandwidth gap between "materialize everything" and "keep a small
// fixed window" is proportionally larger here than for a 1-step method.
//
// Consumer is still a pointwise, non-reduction function of the state
// (tanh(y1)), so there's no reduction to hide the fold behind -- per the
// user's requirement ("assuming the consumer is per-output and not
// reduced").
//
// Build: g++ apps/ode_implicit/ode_bdf3.cpp -O3 -march=native -fopenmp
//        -Iinclude -Lbuild/src -lHalide -lpthread -ldl -o /tmp/odebdf3
//        -std=c++17 ; LD_LIBRARY_PATH=build/src /tmp/odebdf3 [B T mu h K]

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4096;
    int T = argc > 2 ? atoi(argv[2]) : 200001;
    double mu_d = argc > 3 ? atof(argv[3]) : 1000.0;
    double h_d = argc > 4 ? atof(argv[4]) : 1e-5;
    int K = argc > 5 ? atoi(argv[5]) : 3;
    // Expr::Expr(double) is explicit, and mu/h get used throughout in
    // arithmetic with Expr operands, so wrap them once here (an ambiguous
    // overload otherwise, since double doesn't implicitly become Expr like
    // int/float do).
    Expr mu = Expr(mu_d);
    Expr h = Expr(h_d);

    Buffer<double> y1_0(B), y2_0(B);
    srand(7);
    for (int b = 0; b < B; b++) {
        y1_0(b) = 2.0 + (rand() % 100) / 1000.0 - 0.05;
        y2_0(b) = (rand() % 100) / 1000.0 - 0.05;
    }

    // One closed-form Newton step for the generalized BDF residual
    //   F(y) = a*y - rhs - h*f(y) = 0
    // (a=1, rhs=y_n recovers plain backward Euler.)
    auto newton_step = [&](Expr cy1, Expr cy2, Expr a, Expr rhs1, Expr rhs2) -> std::pair<Expr, Expr> {
        Expr f1 = cy2;
        Expr f2 = mu * (Expr(1.0) - cy1 * cy1) * cy2 - cy1;
        Expr F1 = a * cy1 - rhs1 - h * f1;
        Expr F2 = a * cy2 - rhs2 - h * f2;
        Expr J11 = a;
        Expr J12 = -h;
        Expr J21 = h * (Expr(2.0) * mu * cy1 * cy2 + Expr(1.0));
        Expr J22 = a - h * mu * (Expr(1.0) - cy1 * cy1);
        Expr det = J11 * J22 - J12 * J21;
        Expr inv_det = Expr(1.0) / det;
        Expr delta1 = (J22 * F1 - J12 * F2) * inv_det;
        Expr delta2 = (-J21 * F1 + J11 * F2) * inv_det;
        return {cy1 - delta1, cy2 - delta2};
    };
    auto solve = [&](Expr guess1, Expr guess2, Expr a, Expr rhs1, Expr rhs2) -> std::pair<Expr, Expr> {
        Expr cy1 = guess1, cy2 = guess2;
        for (int k = 0; k < K; k++) {
            auto next = newton_step(cy1, cy2, a, rhs1, rhs2);
            cy1 = next.first;
            cy2 = next.second;
        }
        return {cy1, cy2};
    };

    // BDF3 (a, rhs) given the 3-step history (p1=y_{n}, p2=y_{n-1},
    // p3=y_{n-2}); the ONLY formula used in the recursive branch -- order
    // selection for the startup steps happens separately, in a non-recursive
    // bootstrap, so the recursive branch is a single, unconditional formula
    // (required: Halide's inductive base-case proof needs exactly one
    // select() per tuple component, with the self-referencing recursive
    // branch free of any further nested selects).
    auto bdf3 = [&](Expr p1, Expr p2, Expr p3) {
        return std::pair<Expr, Expr>{cast<double>(11.0 / 6.0), 3.0 * p1 - 1.5 * p2 + (1.0 / 3.0) * p3};
    };

    // Non-recursive bootstrap: explicitly computes y(0), y(1), y(2) as plain
    // Exprs directly chained from the initial condition (no Func
    // self-reference at all), then picks the right one by t via nested
    // selects that live entirely OUTSIDE the recursive branch.
    auto bootstrap = [&](Expr b, Expr tt) -> Tuple {
        Expr y0_1 = y1_0(b), y0_2 = y2_0(b);
        auto s1 = solve(y0_1, y0_2, cast<double>(1.0), y0_1, y0_2);            // BDF1 -> y(1)
        auto s2 = solve(s1.first, s1.second, cast<double>(1.5),
                        2.0 * s1.first - 0.5 * y0_1,
                        2.0 * s1.second - 0.5 * y0_2);                         // BDF2 -> y(2)
        Expr r1 = select(tt <= 0, y0_1, select(tt == 1, s1.first, s2.first));
        Expr r2 = select(tt <= 0, y0_2, select(tt == 1, s1.second, s2.second));
        return Tuple(r1, r2);
    };

    auto build = [&](bool inductive) -> Func {
        Var b("b"), t("t");
        Func state(std::vector<Type>{Float(64), Float(64)}, inductive ? "state_ind" : "state_mat");

        if (inductive) {
            Expr p1_1 = state(b, t - 1)[0], p1_2 = state(b, t - 1)[1];
            Expr p2_1 = state(b, t - 2)[0], p2_2 = state(b, t - 2)[1];
            Expr p3_1 = state(b, t - 3)[0], p3_2 = state(b, t - 3)[1];
            auto o1 = bdf3(p1_1, p2_1, p3_1);
            auto o2 = bdf3(p1_2, p2_2, p3_2);
            auto sol = solve(p1_1, p1_2, o1.first, o1.second, o2.second);
            Tuple base = bootstrap(b, t);
            state(b, t) = Tuple(
                select(t <= 2, base[0], likely(sol.first)),
                select(t <= 2, base[1], likely(sol.second)));
        } else {
            Tuple base0 = bootstrap(b, t);
            state(b, t) = base0;
            RDom rt(3, T - 3, "rt");
            Expr p1_1 = state(b, rt - 1)[0], p1_2 = state(b, rt - 1)[1];
            Expr p2_1 = state(b, rt - 2)[0], p2_2 = state(b, rt - 2)[1];
            Expr p3_1 = state(b, rt - 3)[0], p3_2 = state(b, rt - 3)[1];
            auto o1 = bdf3(p1_1, p2_1, p3_1);
            auto o2 = bdf3(p1_2, p2_2, p3_2);
            auto sol = solve(p1_1, p1_2, o1.first, o1.second, o2.second);
            state(b, rt) = Tuple(sol.first, sol.second);
        }

        Func obs("obs");
        obs(b, t) = tanh(state(b, t)[0]);

        const int V = 8;
        Var bo("bo"), bi("bi");
        obs.bound(b, 0, B).bound(t, 0, T)
           .split(b, bo, bi, V).reorder(bi, t, bo).vectorize(bi).parallel(bo);
        if (inductive) {
            state.reorder_storage(b, t)
                 .compute_at(obs, t).store_at(obs, bo).fold_storage(t, 4).vectorize(b, V);
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

        // C++ -O3 streaming reference: only keeps the last 3 states (window
        // reference implementation, same fixed-step/fixed-Newton algorithm).
        std::vector<double> cy((size_t)B * T);
        double tc = 0;
        {
            const double *ya = y1_0.data(), *yb = y2_0.data();
            auto t0 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < B; b++) {
                double h1[4], h2[4];  // ring buffer, index t%4
                h1[0] = ya[b]; h2[0] = yb[b];
                cy[(size_t)b * T] = std::tanh(h1[0]);
                for (int t = 1; t < T; t++) {
                    double p1_1 = h1[(t - 1) & 3], p1_2 = h2[(t - 1) & 3];
                    double p2_1 = h1[std::max(t - 2, 0) & 3], p2_2 = h2[std::max(t - 2, 0) & 3];
                    double p3_1 = h1[std::max(t - 3, 0) & 3], p3_2 = h2[std::max(t - 3, 0) & 3];
                    double a, rhs1, rhs2;
                    if (t <= 1) { a = 1.0; rhs1 = p1_1; rhs2 = p1_2; }
                    else if (t == 2) { a = 1.5; rhs1 = 2 * p1_1 - 0.5 * p2_1; rhs2 = 2 * p1_2 - 0.5 * p2_2; }
                    else { a = 11.0 / 6.0; rhs1 = 3 * p1_1 - 1.5 * p2_1 + p3_1 / 3.0; rhs2 = 3 * p1_2 - 1.5 * p2_2 + p3_2 / 3.0; }
                    double cy1 = p1_1, cy2 = p1_2;
                    for (int k = 0; k < K; k++) {
                        double f1 = cy2;
                        double f2 = mu * (1.0 - cy1 * cy1) * cy2 - cy1;
                        double F1 = a * cy1 - rhs1 - h * f1;
                        double F2 = a * cy2 - rhs2 - h * f2;
                        double J11 = a, J12 = -h;
                        double J21 = h * (2.0 * mu * cy1 * cy2 + 1.0);
                        double J22 = a - h * mu * (1.0 - cy1 * cy1);
                        double det = J11 * J22 - J12 * J21;
                        double inv_det = 1.0 / det;
                        double d1 = (J22 * F1 - J12 * F2) * inv_det;
                        double d2 = (-J21 * F1 + J11 * F2) * inv_det;
                        cy1 -= d1; cy2 -= d2;
                    }
                    h1[t & 3] = cy1; h2[t & 3] = cy2;
                    cy[(size_t)b * T + t] = std::tanh(cy1);
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            tc = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        double err = 0, err_ind_ni = 0, err_ind_ref = 0; bool bad = false;
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) {
            double a = ri(b, t), c = rn(b, t), g = cy[(size_t)b * T + t];
            if (std::isnan(a) || std::isnan(c)) bad = true;
            err = std::max({err, std::abs(a - g), std::abs(c - g)});
            err_ind_ni = std::max(err_ind_ni, std::abs(a - c));
            err_ind_ref = std::max(err_ind_ref, std::abs(a - g));
        }
        printf("  inductive vs non-inductive (Halide-vs-Halide) max diff: %g\n", err_ind_ni);
        printf("  inductive vs C++ reference max diff:                    %g\n", err_ind_ref);
        double traj = (double)2 * B * T * 8 / (1024.0 * 1024.0);
        printf("BDF3 implicit ODE (Van der Pol, K=%d Newton steps, double precision)  B=%d T=%d mu=%.2f h=%.9f  (trajectory %.1f MB)\n",
               K, B, T, mu, h, traj);
        printf("  inductive Halide (fold state->4):  %8.3f ms\n", ti);
        printf("  non-inductive Halide (materialize): %8.3f ms\n", tn);
        printf("  (C++ sanity ref, ignore timing):    %8.3f ms\n", tc);
        printf("  max abs err %.3g%s -> %s\n", err, bad ? " (NaN)" : "", (!bad && err < 1e-2) ? "PASS" : "FAIL");

        auto dump = [](const char *p, const double *d, size_t n) {
            FILE *f = fopen(p, "wb"); if (f) { fwrite(d, sizeof(double), n, f); fclose(f); } };
        dump("apps/ode_implicit/bdf3_y1_0.bin", y1_0.data(), B);
        dump("apps/ode_implicit/bdf3_y2_0.bin", y2_0.data(), B);
        std::vector<double> yflat((size_t)B * T);
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) yflat[(size_t)b * T + t] = ri(b, t);
        dump("apps/ode_implicit/bdf3_obs.bin", yflat.data(), (size_t)B * T);
        FILE *fp = fopen("apps/ode_implicit/bdf3_params.txt", "w");
        if (fp) { fprintf(fp, "%d %d %.9f %.12f %d %.6f\n", B, T, mu, h, K, ti); fclose(fp); }
        return (!bad && err < 1e-2) ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
