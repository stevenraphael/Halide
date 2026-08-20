// 2D SPARSE variant of the Allen-Cahn free-energy observer benchmark.
//
// Same physics dy/dt = eps*A*y + y - y^3, but now on a 2D grid: A is the
// 5-point Laplacian stencil (A y)_{x,y} = y_{x-1,y} + y_{x+1,y} + y_{x,y-1} +
// y_{x,y+1} - 4 y_{x,y}, applied in O(D^2) per step instead of the O(D)
// per-step cost of the 1D version (ode_observer_sparse_test.cpp). This is the
// realistic phase-field / reaction-diffusion regime (grain growth, Allen-Cahn
// on a real grid is 2D/3D, not 1D). Observer = Ginzburg-Landau free energy
// (dissipating Lyapunov functional), now with a 2D gradient term.
//
// Build (USE -O3):
//   g++ apps/ode/ode_observer_sparse_2d_test.cpp -O3 -march=native -Idistrib/include \
//       -Lbuild/src -lHalide -lpthread -ldl -o /tmp/ode_sp_2d -std=c++17
//   LD_LIBRARY_PATH=build/src HL_NUM_THREADS=1 /tmp/ode_sp_2d [D B T]
//
// D is the grid side length (grid has D*D points).

#include "Halide.h"
#include <boost/numeric/odeint.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int D = argc > 1 ? atoi(argv[1]) : 64;  // grid is D x D
    int B = argc > 2 ? atoi(argv[2]) : 1;
    int T = argc > 3 ? atoi(argv[3]) : 2048;
    const float h = 0.02f, eps = 0.1f;

    Buffer<float> y0(D, D, B);
    srand(5);
    for (int bb = 0; bb < B; bb++)
        for (int yy = 0; yy < D; yy++)
            for (int xx = 0; xx < D; xx++) y0(xx, yy, bb) = (float)(rand() % 200) / 100.0f - 1.0f;

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

    Var x("x"), y("y"), b("b"), n("n");
    // 2D Ginzburg-Landau free energy density: 0.5*eps*|grad y|^2 + double well.
    auto energy_density = [&](Func yf, RVar rx, RVar ry, Expr time) {
        Expr yc = yf(rx, ry, b, time);
        Expr yxp = yf(clamp(cast<int>(rx) + 1, 0, D - 1), ry, b, time);
        Expr yyp = yf(rx, clamp(cast<int>(ry) + 1, 0, D - 1), b, time);
        Expr gx = yxp - yc, gy = yyp - yc;
        return 0.5f * eps * (gx * gx + gy * gy) + 0.25f * (1 - yc * yc) * (1 - yc * yc);
    };

    try {
        // ---------- 1. Inductive dynamics (5-point stencil) + observer, state folded ----------
        Func y_i(Float(32), "y_i"), E_i("E_i");
        {
            y_i(x, y, b, n) = cast<float>(0);
            Expr nm1 = n - 1, nm2 = n - 2;
            // r.x,r.y = output grid coords, r.z = stencil tap {0=center,1=x-1,2=x+1,3=y-1,4=y+1}.
            // All are RVars so dim 0/1 never hold the pure var in shifted position -- only
            // n recurses. Work per step is 5*D*D = O(D^2).
            RDom r(0, D, 0, D, 0, 5, "r");
            Expr ix = r.x, iy = r.y, off = r.z;
            Expr nbx = clamp(select(off == 1, ix - 1, select(off == 2, ix + 1, ix)), 0, D - 1);
            Expr nby = clamp(select(off == 3, iy - 1, select(off == 4, iy + 1, iy)), 0, D - 1);
            Expr w = select(off == 0, -4.0f, 1.0f);  // 5-point Laplacian stencil weights
            // Pointwise part (carry + AB2 reaction g(y)=y-y^3), gated to off==0 (once/pixel).
            Expr c1 = y_i(ix, iy, b, nm1), c2 = y_i(ix, iy, b, nm2);
            Expr react = 1.5f * (c1 - c1 * c1 * c1) - 0.5f * (c2 - c2 * c2 * c2);
            Expr onceper = c1 + h * react;
            // Diffusion eps*A*y accumulated over the 5-point stencil (AB2-weighted).
            Expr diff = h * eps * w * (1.5f * y_i(nbx, nby, b, nm1) - 0.5f * y_i(nbx, nby, b, nm2));
            // Seed at off==0 by ASSIGNMENT (not accumulate-onto-zero) so the folded slice
            // needs no per-step re-init; off>0 accumulate. off is the outer (unrolled) loop.
            y_i(ix, iy, b, n) = select(n <= 0, y0(ix, iy, b),
                                       select(off == 0, onceper + diff, y_i(ix, iy, b, n) + diff));

            RDom rd(0, D, 0, D, "rd");
            E_i(b, n) = cast<float>(0);
            E_i(b, n) += energy_density(y_i, rd.x, rd.y, n);
            E_i.compute_root();
            E_i.update(0).reorder(rd.x, rd.y, b, n);
            y_i.compute_at(E_i, n).store_root().fold_storage(n, 3);
            y_i.update(0).allow_race_conditions().unroll(r.z).vectorize(r.x, 16);
            E_i.bound(b, 0, B).bound(n, 0, T);
            E_i.compile_jit();
        }

        // ---------- 2. Non-inductive: MATERIALIZE full trajectory, then reduce ----------
        Func y_m(Float(32), "y_m"), E_m("E_m");
        {
            RDom rn(1, T - 1, "rn");
            y_m(x, y, b, n) = undef<float>();
            y_m(x, y, b, 0) = y0(x, y, b);
            Expr p1 = rn - 1, p2 = clamp(rn - 2, 0, T - 1);
            auto lap = [&](Expr t) {
                return y_m(clamp(x - 1, 0, D - 1), y, b, t) + y_m(clamp(x + 1, 0, D - 1), y, b, t) +
                       y_m(x, clamp(y - 1, 0, D - 1), b, t) + y_m(x, clamp(y + 1, 0, D - 1), b, t) -
                       4.0f * y_m(x, y, b, t);
            };
            auto f = [&](Expr t) { Expr c = y_m(x, y, b, t); return eps * lap(t) + c - c * c * c; };
            y_m(x, y, b, rn) = y_m(x, y, b, p1) + h * (1.5f * f(p1) - 0.5f * f(p2));

            RDom rd(0, D, 0, D, "rd");
            E_m(b, n) = cast<float>(0);
            E_m(b, n) += energy_density(y_m, rd.x, rd.y, n);
            y_m.compute_root();  // full D*D*B*T
            y_m.update(0).unscheduled();
            y_m.update(1).reorder(x, y, b, rn).vectorize(x, 16);
            E_m.compute_root();
            E_m.update(0).reorder(rd.x, rd.y, b, n);
            E_m.bound(b, 0, B).bound(n, 0, T);
            E_m.compile_jit();
        }

        Buffer<float> ei(B, T), em(B, T);
        E_i.realize(ei);
        E_m.realize(em);
        double t_i = bench(5, [&]() { E_i.realize(ei); });
        double t_m = bench(5, [&]() { E_m.realize(em); });

        int N = D * D;
        auto idx = [&](int xx, int yy) { return yy * D + xx; };
        auto energy = [&](const std::vector<float> &yv) {
            double e = 0;
            for (int yy = 0; yy < D; yy++)
                for (int xx = 0; xx < D; xx++) {
                    float c = yv[idx(xx, yy)];
                    float gx = (xx + 1 < D) ? (yv[idx(xx + 1, yy)] - c) : 0.f;
                    float gy = (yy + 1 < D) ? (yv[idx(xx, yy + 1)] - c) : 0.f;
                    e += 0.5 * eps * (gx * gx + gy * gy) + 0.25 * (1 - (double)c * c) * (1 - (double)c * c);
                }
            return (float)e;
        };
        auto rhs = [&](const std::vector<float> &yv, std::vector<float> &f) {
            for (int yy = 0; yy < D; yy++)
                for (int xx = 0; xx < D; xx++) {
                    int xm = xx > 0 ? xx - 1 : 0, xp = xx + 1 < D ? xx + 1 : D - 1;
                    int ym = yy > 0 ? yy - 1 : 0, yp = yy + 1 < D ? yy + 1 : D - 1;
                    float c = yv[idx(xx, yy)];
                    float lap = yv[idx(xm, yy)] + yv[idx(xp, yy)] + yv[idx(xx, ym)] + yv[idx(xx, yp)] - 4.f * c;
                    f[idx(xx, yy)] = eps * lap + c - c * c * c;
                }
        };

        // ---------- 3. Boost.odeint + observer ----------
        namespace odeint = boost::numeric::odeint;
        using state = std::vector<float>;
        auto sys = [&](const state &yv, state &dy, double) { rhs(yv, dy); };
        std::vector<float> eb((size_t)B * T);
        double t_boost = bench(5, [&]() {
            for (int bb = 0; bb < B; bb++) {
                state xv(N);
                for (int yy = 0; yy < D; yy++)
                    for (int xx = 0; xx < D; xx++) xv[idx(xx, yy)] = y0(xx, yy, bb);
                int step = 0;
                eb[(size_t)bb * T + step++] = energy(xv);
                odeint::adams_bashforth<2, state> ab;
                double t = 0, dt = h;
                ab.initialize(odeint::euler<state>(), sys, xv, t, dt);
                eb[(size_t)bb * T + step++] = energy(xv);
                for (int nn = 2; nn < T; nn++) {
                    ab.do_step(sys, xv, t, dt);
                    t += dt;
                    eb[(size_t)bb * T + step++] = energy(xv);
                }
            }
        });

        // ---------- 4. C++ reference + observer ----------
        std::vector<float> ec((size_t)B * T);
        double t_cpp = bench(5, [&]() {
            std::vector<float> yv(N), f1(N), f2(N), tmp(N);
            for (int bb = 0; bb < B; bb++) {
                for (int yy = 0; yy < D; yy++)
                    for (int xx = 0; xx < D; xx++) yv[idx(xx, yy)] = y0(xx, yy, bb);
                rhs(yv, f1); f2 = f1;
                ec[(size_t)bb * T + 0] = energy(yv);
                for (int nn = 1; nn < T; nn++) {
                    for (int i = 0; i < N; i++) tmp[i] = yv[i] + h * (1.5f * f1[i] - 0.5f * f2[i]);
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
        double err_i = relerr([&](int bb, int nn) { return ei(bb, nn); });
        double err_m = relerr([&](int bb, int nn) { return em(bb, nn); });
        double err_b = relerr([&](int bb, int nn) { return eb[(size_t)bb * T + nn]; });

        double fold_kb = (double)N * B * 3 * sizeof(float) / 1024.0;
        double full_kb = (double)N * B * T * sizeof(float) / 1024.0;
        printf("2D SPARSE Allen-Cahn free-energy observer  grid=%dx%d B=%d T=%d, single core\n", D, D, B, T);
        printf("  Halide inductive + observer:  %8.3f ms  (rel err %.2g)  state %.1f KB (D^2*B*3)\n", t_i, err_i, fold_kb);
        printf("  Halide non-ind (materialize): %8.3f ms  (rel err %.2g)  state %.1f KB (D^2*B*T)\n", t_m, err_m, full_kb);
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
