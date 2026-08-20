// Follow-up to ode_observer_sparse_test.cpp: try to close the last ~10% gap between the
// (already stencil-fused) Halide inductive version and hand-written Boost/C++ on the
// SPARSE Allen-Cahn free-energy-observer problem.
//
// Hypothesis: after unroll(r.y) fused the stencil, the remaining overhead is the OBSERVER
// reduction E(b,n) += energy_density(slice) running SCALAR over the D components while the
// producer is 16-wide. Here we compare:
//   A. inductive, scalar observer   (the current best from ode_observer_sparse_test.cpp)
//   B. inductive, VECTORIZED observer (atomic().vectorize on the reduction)
//   C. Boost.odeint + observer
//   D. C++ reference + observer
//
// Build: g++ apps/ode/ode_observer_sparse_fused_test.cpp -O3 -march=native \
//   -Idistrib/include -Lbuild/src -lHalide -lpthread -ldl -o /tmp/ode_spf -std=c++17
//   LD_LIBRARY_PATH=build/src HL_NUM_THREADS=1 /tmp/ode_spf [D B T]

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
        for (int i = 0; i < D; i++) y0(i, bb) = (float)(rand() % 200) / 100.0f - 1.0f;

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

    // Stencil RHS f(v)_i = eps*(v_{i-1}-2v_i+v_{i+1}) + v_i - v_i^3 as a Func of a 2-D Func.
    auto rhs_of = [&](Func v, const std::string &nm) {
        Func f(nm);
        f(d, b) = eps * (v(clamp(d - 1, 0, D - 1), b) - 2.0f * v(d, b) + v(clamp(d + 1, 0, D - 1), b)) +
                  v(d, b) - v(d, b) * v(d, b) * v(d, b);
        f.compute_root();
        return f;
    };
    // One classical RK4 step y0 -> y1 (matches Boost's runge_kutta4 startup exactly).
    // Computed once (compute_root); negligible vs the T-step integration.
    auto make_y1 = [&](const std::string &tag) {
        Func Y0("Y0_" + tag);
        Y0(d, b) = y0(d, b);
        Y0.compute_root();
        Func k1 = rhs_of(Y0, "k1_" + tag);
        Func s2("s2_" + tag); s2(d, b) = Y0(d, b) + (0.5f * h) * k1(d, b); s2.compute_root();
        Func k2 = rhs_of(s2, "k2_" + tag);
        Func s3("s3_" + tag); s3(d, b) = Y0(d, b) + (0.5f * h) * k2(d, b); s3.compute_root();
        Func k3 = rhs_of(s3, "k3_" + tag);
        Func s4("s4_" + tag); s4(d, b) = Y0(d, b) + h * k3(d, b); s4.compute_root();
        Func k4 = rhs_of(s4, "k4_" + tag);
        Func y1("y1_" + tag);
        y1(d, b) = Y0(d, b) + (h / 6.0f) * (k1(d, b) + 2.0f * k2(d, b) + 2.0f * k3(d, b) + k4(d, b));
        y1.compute_root();
        return y1;
    };

    // Build one inductive (dynamics) + observer pipeline; `vec_obs` toggles observer SIMD.
    auto build = [&](const char *tag, bool vec_obs) {
        Func y1 = make_y1(std::string("ind_") + tag);
        Func y(Float(32), std::string("y_") + tag), E(std::string("E_") + tag);
        y(d, b, n) = cast<float>(0);
        Expr nm1 = n - 1, nm2 = n - 2;
        RDom r(0, D, "r");
        Expr i = r.x;
        Expr iL = clamp(i - 1, 0, D - 1), iR = clamp(i + 1, 0, D - 1);
        Expr c1 = y(i, b, nm1), c2 = y(i, b, nm2);
        // Build the full RHS the SAME way as the non-inductive form and the C++
        // reference: f = eps*(yL - 2y + yR) + (c - c^3), then one h-scaled AB2
        // combine. This matches the reference's float32 op-grouping (the previous
        // per-offset scatter reassociated the diffusion sum and drifted ~1e-6).
        auto fexpr = [&](Expr c, Expr t) {
            Expr lap = y(iL, b, t) - 2.0f * y(i, b, t) + y(iR, b, t);
            // Left-associate exactly like the non-inductive form (:148) and the
            // C++ reference (:181): (eps*lap + c) - c^3, NOT eps*lap + (c - c^3).
            // In float32 these are different reassociations; matching the grouping
            // makes inductive and non-inductive bit-identical.
            return eps * lap + c - c * c * c;
        };
        Expr fP1 = fexpr(c1, nm1), fP2 = fexpr(c2, nm2);
        // Base cases: n<=0 -> y0, n==1 -> RK4 step y1; AB2 recursion (self-refs at
        // n-1,n-2) in the outer select's false branch for n>=2.
        Expr base = select(n <= 0, y0(i, b), y1(i, b));
        y(i, b, n) = select(n <= 1, base, c1 + h * (1.5f * fP1 - 0.5f * fP2));

        // Realistic (cheap, untuned) observer: the mean order parameter <y> = (1/D) sum_i y_i,
        // the standard phase-field monitoring diagnostic. Just an add per element -- so the
        // observer arithmetic is negligible and any inductive-vs-materialize gap is purely
        // the L1-vs-DRAM cost of reading each slice.
        RDom rd(0, D, "rd");
        E(b, n) = cast<float>(0);
        E(b, n) += y(rd.x, b, n) * (1.0f / D);

        E.compute_root();
        if (vec_obs) {
            // Vectorize the associative sum reduction across components.
            E.update(0).atomic().reorder(rd.x, b, n).vectorize(rd.x, 16);
        } else {
            E.update(0).reorder(rd.x, b, n);
        }
        y.compute_at(E, n).store_root().fold_storage(n, 3);
        // Reads are all from prior slices (n-1, n-2), so vectorizing r.x is race-free
        // despite the self-reference; Halide's conservative check needs the override.
        y.update(0).allow_race_conditions().vectorize(r.x, 16);
        E.bound(b, 0, B).bound(n, 0, T);
        E.compile_jit();
        return E;
    };

    Func E_scalar, E_vec;
    try {
        E_scalar = build("scalar", false);
        E_vec = build("vec", true);
    } catch (const Halide::Error &e) {
        printf("BUILD ERROR: %s\n", e.what());
        return 3;
    }

    // Non-inductive (materialize full trajectory) with the SAME vectorized observer.
    Func E_mat("E_mat");
    {
        Func y1 = make_y1("mat");
        Func y_m(Float(32), "y_m");
        RDom rn(2, T - 2, "rn");  // slices 0,1 seeded; AB2 scan from n=2
        y_m(d, b, n) = undef<float>();
        y_m(d, b, 0) = y0(d, b);
        y_m(d, b, 1) = y1(d, b);  // RK4 startup
        Expr p1 = rn - 1, p2 = rn - 2;
        auto lap = [&](Expr t) {
            return y_m(clamp(d - 1, 0, D - 1), b, t) - 2.0f * y_m(d, b, t) +
                   y_m(clamp(d + 1, 0, D - 1), b, t);
        };
        auto f = [&](Expr t) { Expr c = y_m(d, b, t); return eps * lap(t) + c - c * c * c; };
        y_m(d, b, rn) = y_m(d, b, p1) + h * (1.5f * f(p1) - 0.5f * f(p2));

        RDom rd(0, D, "rd");
        E_mat(b, n) = cast<float>(0);
        E_mat(b, n) += y_m(rd.x, b, n) * (1.0f / D);
        y_m.compute_root();
        y_m.update(0).unscheduled();  // slice 0 init
        y_m.update(1).unscheduled();  // slice 1 (RK4) init
        y_m.update(2).reorder(d, b, rn).vectorize(d, 16);  // AB2 scan
        E_mat.compute_root();
        E_mat.update(0).atomic().reorder(rd.x, b, n).vectorize(rd.x, 16);  // same vectorized observer
        E_mat.bound(b, 0, B).bound(n, 0, T);
        E_mat.compile_jit();
    }

    Buffer<float> es(B, T), ev(B, T), emat(B, T);
    E_scalar.realize(es);
    E_vec.realize(ev);
    E_mat.realize(emat);
    double t_scalar = bench(5, [&]() { E_scalar.realize(es); });
    double t_vec = bench(5, [&]() { E_vec.realize(ev); });
    double t_mat = bench(5, [&]() { E_mat.realize(emat); });

    // energy + rhs helpers
    auto energy = [&](const std::vector<float> &y) {  // mean order parameter <y>
        double e = 0;
        for (int i = 0; i < D; i++) e += y[i];
        return (float)(e / D);
    };
    auto rhs = [&](const std::vector<float> &y, std::vector<float> &f) {
        for (int i = 0; i < D; i++) {
            float lm = y[i > 0 ? i - 1 : 0], lp = y[i + 1 < D ? i + 1 : D - 1];
            f[i] = eps * (lm - 2.f * y[i] + lp) + y[i] - y[i] * y[i] * y[i];
        }
    };

    namespace odeint = boost::numeric::odeint;
    using state = std::vector<float>;
    auto sys = [&](const state &y, state &dy, double) { rhs(y, dy); };
    // One RK4 step (matches make_y1 / Boost's runge_kutta4 startup).
    auto rk4_step = [&](state &y) {
        state k1(D), k2(D), k3(D), k4(D), tmp(D);
        rhs(y, k1);
        for (int i = 0; i < D; i++) tmp[i] = y[i] + 0.5f * h * k1[i]; rhs(tmp, k2);
        for (int i = 0; i < D; i++) tmp[i] = y[i] + 0.5f * h * k2[i]; rhs(tmp, k3);
        for (int i = 0; i < D; i++) tmp[i] = y[i] + h * k3[i]; rhs(tmp, k4);
        for (int i = 0; i < D; i++) y[i] += (h / 6.f) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    };

    // Idiomatic Boost.odeint: integrate_n_steps + observer, with runge_kutta4 as the
    // initializing stepper so its startup matches make_y1 exactly (tight agreement).
    typedef odeint::adams_bashforth<2, state, double, state, double, odeint::range_algebra,
                                    odeint::default_operations, odeint::initially_resizer,
                                    odeint::runge_kutta4<state>>
        ab2_rk4;
    std::vector<float> eb((size_t)B * T);
    double t_boost = bench(5, [&]() {
        for (int bb = 0; bb < B; bb++) {
            state x(D);
            for (int i = 0; i < D; i++) x[i] = y0(i, bb);
            int step = 0;
            auto obs = [&](const state &xs, double) {
                if (step < T) eb[(size_t)bb * T + step++] = energy(xs);
            };
            ab2_rk4 ab;
            odeint::integrate_n_steps(ab, sys, x, 0.0, (double)h, T - 1, obs);
        }
    });

    std::vector<float> ec((size_t)B * T);
    double t_cpp = bench(5, [&]() {
        std::vector<float> yv(D), f1(D), f2(D), tmp(D);
        for (int bb = 0; bb < B; bb++) {
            for (int i = 0; i < D; i++) yv[i] = y0(i, bb);
            ec[(size_t)bb * T + 0] = energy(yv);  // n=0
            state y0v = yv;
            rk4_step(yv);                          // n=1 via RK4
            ec[(size_t)bb * T + 1] = energy(yv);
            rhs(y0v, f2);                          // f_0
            rhs(yv, f1);                           // f_1
            for (int nn = 2; nn < T; nn++) {       // AB2 from n=2
                for (int i = 0; i < D; i++) tmp[i] = yv[i] + h * (1.5f * f1[i] - 0.5f * f2[i]);
                yv = tmp; f2 = f1; rhs(yv, f1);
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
    double err_s = relerr([&](int bb, int nn) { return es(bb, nn); });
    double err_v = relerr([&](int bb, int nn) { return ev(bb, nn); });
    double err_m = relerr([&](int bb, int nn) { return emat(bb, nn); });
    double err_b = relerr([&](int bb, int nn) { return eb[(size_t)bb * T + nn]; });

    printf("SPARSE observer, closing the gap  D=%d B=%d T=%d, single core\n", D, B, T);
    printf("  A. inductive,   scalar observer:      %7.3f ms  (rel err %.2g)  state %.0f KB\n", t_scalar, err_s, (double)D * B * 3 * 4 / 1024);
    printf("  B. inductive,   vectorized observer:  %7.3f ms  (rel err %.2g)  state %.0f KB\n", t_vec, err_v, (double)D * B * 3 * 4 / 1024);
    printf("  E. non-ind mat, vectorized observer:  %7.3f ms  (rel err %.2g)  state %.0f KB\n", t_mat, err_m, (double)D * B * T * 4 / 1024);
    printf("  C. Boost.odeint (integrate_n_steps + rk4 init + observer): %7.3f ms  (rel err %.2g)\n", t_boost, err_b);
    printf("  D. C++ reference + observer:          %7.3f ms\n", t_cpp);
    printf("  gap B vs Boost: %+.1f%%   inductive(B) vs non-ind(E): %.2fx\n",
           100.0 * (t_vec - t_boost) / t_boost, t_mat / t_vec);
    // All four now use an identical RK4 startup, so all should match the reference tightly.
    bool pass = err_s < 1e-3 && err_v < 1e-3 && err_m < 1e-3 && err_b < 1e-3;
    printf("  -> %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
