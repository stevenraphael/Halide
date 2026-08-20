// Fixed-step, fixed-Newton-iteration BDF3 on Robertson's classic stiff
// chemical-kinetics test problem ("ROBER", Hairer & Wanner):
//   y1' = -k1*y1 + k3*y2*y3
//   y2' =  k1*y1 - k3*y2*y3 - k2*y2^2
//   y3' =  k2*y2^2
// with k1=0.04, k2=3e7, k3=1e4, y0=(1,0,0), t in [0, 40] (a standard
// truncated interval for this problem; the full literature version runs to
// t=1e11 on a log scale, which -- like Van der Pol's pathological initial
// step -- would need an enormous fixed-step dynamic range and isn't a
// reasonable target for a fixed-step method).
//
// 3-dimensional state (unlike ode_bdf3.cpp's 2D Van der Pol), so the Newton
// solve is a closed-form 3x3 linear solve (Cramer's rule / adjugate) instead
// of 2x2, but otherwise identical structure: same BDF3 formula, same
// non-recursive 2-step bootstrap (BDF1 then BDF2) for startup, same
// fold_storage(t,4) vs materializing-RDom comparison, same pointwise
// (non-reduction) consumer.
//
// Build: g++ apps/ode_implicit/ode_robertson_bdf3.cpp -O3 -march=native
//        -fopenmp -Iinclude -Lbuild/src -lHalide -lpthread -ldl
//        -o /tmp/oderob -std=c++17
//        ; LD_LIBRARY_PATH=build/src /tmp/oderob [B T tfinal K]

#include "Halide.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

struct Vec3 { Expr x, y, z; };

int main(int argc, char **argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4096;
    int T = argc > 2 ? atoi(argv[2]) : 400001;
    double t_final_d = argc > 3 ? atof(argv[3]) : 40.0;
    int K = argc > 4 ? atoi(argv[4]) : 3;
    double h_d = t_final_d / (T - 1);

    const double k1_d = 0.04, k2_d = 3e7, k3_d = 1e4;
    Expr k1 = Expr(k1_d), k2 = Expr(k2_d), k3 = Expr(k3_d);
    Expr h = Expr(h_d);

    // Batch of trajectories clustered around the canonical Robertson initial
    // condition (1,0,0), perturbed slightly in y1/y3 (keeping y2=0, since a
    // nonzero y2 perturbation would violate the y1+y2+y3=1 mass-conservation
    // structure the problem relies on) so the B trajectories are distinct.
    Buffer<double> y1_0(B), y2_0(B), y3_0(B);
    srand(11);
    for (int b = 0; b < B; b++) {
        double d = (rand() % 100) / 100000.0 - 0.0005;  // +/- 5e-4
        y1_0(b) = 1.0 + d;
        y2_0(b) = 0.0;
        y3_0(b) = -d;
    }

    auto rhs = [&](Vec3 y) -> Vec3 {
        Expr f1 = -k1 * y.x + k3 * y.y * y.z;
        Expr f2 = k1 * y.x - k3 * y.y * y.z - k2 * y.y * y.y;
        Expr f3 = k2 * y.y * y.y;
        return {f1, f2, f3};
    };
    // Analytic Jacobian df_i/dy_j.
    auto jac = [&](Vec3 y) -> std::array<std::array<Expr, 3>, 3> {
        return {{
            {-k1, k3 * y.z, k3 * y.y},
            {k1, -k3 * y.z - Expr(2.0) * k2 * y.y, -k3 * y.y},
            {Expr(0.0), Expr(2.0) * k2 * y.y, Expr(0.0)},
        }};
    };

    // One closed-form Newton step for the generalized BDF residual
    //   F(y) = a*y - rhs - h*f(y) = 0   (vector; a=1,rhs=y_n => backward Euler)
    // solved via a closed-form 3x3 linear solve (Cramer's rule / adjugate).
    auto newton_step = [&](Vec3 y, Expr a, Vec3 rhs_v) -> Vec3 {
        Vec3 f = rhs(y);
        Expr F0 = a * y.x - rhs_v.x - h * f.x;
        Expr F1 = a * y.y - rhs_v.y - h * f.y;
        Expr F2 = a * y.z - rhs_v.z - h * f.z;
        auto Jf = jac(y);
        // J = a*I - h*Jf
        Expr J00 = a - h * Jf[0][0], J01 = -h * Jf[0][1], J02 = -h * Jf[0][2];
        Expr J10 = -h * Jf[1][0], J11 = a - h * Jf[1][1], J12 = -h * Jf[1][2];
        Expr J20 = -h * Jf[2][0], J21 = -h * Jf[2][1], J22 = a - h * Jf[2][2];
        Expr det = J00 * (J11 * J22 - J12 * J21)
                 - J01 * (J10 * J22 - J12 * J20)
                 + J02 * (J10 * J21 - J11 * J20);
        Expr inv_det = Expr(1.0) / det;
        // adjugate (transpose of cofactors) applied to (F0,F1,F2):
        Expr d0 = ((J11 * J22 - J12 * J21) * F0 + (J02 * J21 - J01 * J22) * F1 + (J01 * J12 - J02 * J11) * F2) * inv_det;
        Expr d1 = ((J12 * J20 - J10 * J22) * F0 + (J00 * J22 - J02 * J20) * F1 + (J02 * J10 - J00 * J12) * F2) * inv_det;
        Expr d2 = ((J10 * J21 - J11 * J20) * F0 + (J01 * J20 - J00 * J21) * F1 + (J00 * J11 - J01 * J10) * F2) * inv_det;
        return {y.x - d0, y.y - d1, y.z - d2};
    };
    auto solve = [&](Vec3 guess, Expr a, Vec3 rhs_v) -> Vec3 {
        Vec3 y = guess;
        for (int k = 0; k < K; k++) y = newton_step(y, a, rhs_v);
        return y;
    };

    // BDF3 (a, rhs) given 3-step history p1=y_n, p2=y_{n-1}, p3=y_{n-2}.
    auto bdf3 = [&](Vec3 p1, Vec3 p2, Vec3 p3) -> std::pair<Expr, Vec3> {
        Expr a = Expr(11.0 / 6.0);
        Vec3 rhs_v{
            Expr(3.0) * p1.x - Expr(1.5) * p2.x + Expr(1.0 / 3.0) * p3.x,
            Expr(3.0) * p1.y - Expr(1.5) * p2.y + Expr(1.0 / 3.0) * p3.y,
            Expr(3.0) * p1.z - Expr(1.5) * p2.z + Expr(1.0 / 3.0) * p3.z,
        };
        return {a, rhs_v};
    };

    // Non-recursive bootstrap for y(0), y(1), y(2) (BDF1 then BDF2), same
    // pattern as ode_bdf3.cpp: entirely outside the recursive branch.
    auto bootstrap = [&](Expr b, Expr tt) -> Tuple {
        Vec3 y0{y1_0(b), y2_0(b), y3_0(b)};
        Vec3 s1 = solve(y0, Expr(1.0), y0);                                    // BDF1 -> y(1)
        Vec3 rhs2{Expr(2.0) * s1.x - Expr(0.5) * y0.x,
                  Expr(2.0) * s1.y - Expr(0.5) * y0.y,
                  Expr(2.0) * s1.z - Expr(0.5) * y0.z};
        Vec3 s2 = solve(s1, Expr(1.5), rhs2);                                  // BDF2 -> y(2)
        Expr r0 = select(tt <= 0, y0.x, select(tt == 1, s1.x, s2.x));
        Expr r1 = select(tt <= 0, y0.y, select(tt == 1, s1.y, s2.y));
        Expr r2 = select(tt <= 0, y0.z, select(tt == 1, s1.z, s2.z));
        return Tuple(r0, r1, r2);
    };

    auto build = [&](bool inductive) -> Func {
        Var b("b"), t("t");
        Func state(std::vector<Type>{Float(64), Float(64), Float(64)}, inductive ? "state_ind" : "state_mat");

        if (inductive) {
            Vec3 p1{state(b, t - 1)[0], state(b, t - 1)[1], state(b, t - 1)[2]};
            Vec3 p2{state(b, t - 2)[0], state(b, t - 2)[1], state(b, t - 2)[2]};
            Vec3 p3{state(b, t - 3)[0], state(b, t - 3)[1], state(b, t - 3)[2]};
            auto ar = bdf3(p1, p2, p3);
            Vec3 sol = solve(p1, ar.first, ar.second);
            Tuple base = bootstrap(b, t);
            state(b, t) = Tuple(
                select(t <= 2, base[0], likely(sol.x)),
                select(t <= 2, base[1], likely(sol.y)),
                select(t <= 2, base[2], likely(sol.z)));
        } else {
            state(b, t) = bootstrap(b, t);
            RDom rt(3, T - 3, "rt");
            Vec3 p1{state(b, rt - 1)[0], state(b, rt - 1)[1], state(b, rt - 1)[2]};
            Vec3 p2{state(b, rt - 2)[0], state(b, rt - 2)[1], state(b, rt - 2)[2]};
            Vec3 p3{state(b, rt - 3)[0], state(b, rt - 3)[1], state(b, rt - 3)[2]};
            auto ar = bdf3(p1, p2, p3);
            Vec3 sol = solve(p1, ar.first, ar.second);
            state(b, rt) = Tuple(sol.x, sol.y, sol.z);
        }

        // Pointwise (non-reduction) consumer: y3 grows monotonically from 0
        // toward 1 as the fast species equilibrate; tanh keeps it bounded
        // and nonlinear like the Van der Pol benchmark's consumer.
        Func obs("obs");
        obs(b, t) = tanh(state(b, t)[2]);

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

        // C++ -O3 streaming reference: only keeps the last 3 states.
        std::vector<double> cy((size_t)B * T);
        double tc = 0;
        {
            const double *ya = y1_0.data(), *yb = y2_0.data(), *yc = y3_0.data();
            auto t0 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel for schedule(static)
            for (int b = 0; b < B; b++) {
                double h1x[4], h1y[4], h1z[4];
                h1x[0] = ya[b]; h1y[0] = yb[b]; h1z[0] = yc[b];
                cy[(size_t)b * T] = std::tanh(h1z[0]);
                for (int t = 1; t < T; t++) {
                    int i1 = (t - 1) & 3, i2 = std::max(t - 2, 0) & 3, i3 = std::max(t - 3, 0) & 3;
                    double p1x = h1x[i1], p1y = h1y[i1], p1z = h1z[i1];
                    double p2x = h1x[i2], p2y = h1y[i2], p2z = h1z[i2];
                    double p3x = h1x[i3], p3y = h1y[i3], p3z = h1z[i3];
                    double a, rhsx, rhsy, rhsz;
                    if (t <= 1) { a = 1.0; rhsx = p1x; rhsy = p1y; rhsz = p1z; }
                    else if (t == 2) { a = 1.5; rhsx = 2 * p1x - 0.5 * p2x; rhsy = 2 * p1y - 0.5 * p2y; rhsz = 2 * p1z - 0.5 * p2z; }
                    else { a = 11.0 / 6.0; rhsx = 3 * p1x - 1.5 * p2x + p3x / 3.0; rhsy = 3 * p1y - 1.5 * p2y + p3y / 3.0; rhsz = 3 * p1z - 1.5 * p2z + p3z / 3.0; }
                    double yx = p1x, yy = p1y, yz = p1z;
                    for (int k = 0; k < K; k++) {
                        double f0 = -k1_d * yx + k3_d * yy * yz;
                        double f1 = k1_d * yx - k3_d * yy * yz - k2_d * yy * yy;
                        double f2 = k2_d * yy * yy;
                        double F0 = a * yx - rhsx - h_d * f0;
                        double F1 = a * yy - rhsy - h_d * f1;
                        double F2 = a * yz - rhsz - h_d * f2;
                        double J00 = a - h_d * (-k1_d), J01 = -h_d * (k3_d * yz), J02 = -h_d * (k3_d * yy);
                        double J10 = -h_d * (k1_d), J11 = a - h_d * (-k3_d * yz - 2 * k2_d * yy), J12 = -h_d * (-k3_d * yy);
                        double J20 = -h_d * (0.0), J21 = -h_d * (2 * k2_d * yy), J22 = a - h_d * (0.0);
                        double det = J00 * (J11 * J22 - J12 * J21) - J01 * (J10 * J22 - J12 * J20) + J02 * (J10 * J21 - J11 * J20);
                        double inv_det = 1.0 / det;
                        double d0 = ((J11 * J22 - J12 * J21) * F0 + (J02 * J21 - J01 * J22) * F1 + (J01 * J12 - J02 * J11) * F2) * inv_det;
                        double d1 = ((J12 * J20 - J10 * J22) * F0 + (J00 * J22 - J02 * J20) * F1 + (J02 * J10 - J00 * J12) * F2) * inv_det;
                        double d2 = ((J10 * J21 - J11 * J20) * F0 + (J01 * J20 - J00 * J21) * F1 + (J00 * J11 - J01 * J10) * F2) * inv_det;
                        yx -= d0; yy -= d1; yz -= d2;
                    }
                    h1x[t & 3] = yx; h1y[t & 3] = yy; h1z[t & 3] = yz;
                    cy[(size_t)b * T + t] = std::tanh(yz);
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
        double traj = (double)3 * B * T * 8 / (1024.0 * 1024.0);
        printf("Robertson BDF3 (K=%d Newton steps, double precision)  B=%d T=%d t_final=%.2f h=%.9f  (trajectory %.1f MB)\n",
               K, B, T, t_final_d, h_d, traj);
        printf("  inductive Halide (fold state->4):  %8.3f ms\n", ti);
        printf("  non-inductive Halide (materialize): %8.3f ms\n", tn);
        printf("  (C++ sanity ref, ignore timing):    %8.3f ms\n", tc);
        printf("  max abs err %.3g%s -> %s\n", err, bad ? " (NaN)" : "", (!bad && err < 1e-2) ? "PASS" : "FAIL");

        auto dump = [](const char *p, const double *d, size_t n) {
            FILE *f = fopen(p, "wb"); if (f) { fwrite(d, sizeof(double), n, f); fclose(f); } };
        dump("apps/ode_implicit/rob_y1_0.bin", y1_0.data(), B);
        dump("apps/ode_implicit/rob_y2_0.bin", y2_0.data(), B);
        dump("apps/ode_implicit/rob_y3_0.bin", y3_0.data(), B);
        std::vector<double> yflat((size_t)B * T);
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) yflat[(size_t)b * T + t] = ri(b, t);
        dump("apps/ode_implicit/rob_obs.bin", yflat.data(), (size_t)B * T);
        FILE *fp = fopen("apps/ode_implicit/rob_params.txt", "w");
        if (fp) { fprintf(fp, "%d %d %.9f %.12f %d %.6f\n", B, T, t_final_d, h_d, K, ti); fclose(fp); }
        return (!bad && err < 1e-2) ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
