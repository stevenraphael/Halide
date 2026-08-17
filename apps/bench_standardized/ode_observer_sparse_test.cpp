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

#include "../support/bench_harness.h"
#include "Halide.h"
#include <boost/numeric/odeint.hpp>
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

    // Standardized protocol (warmup + trials + median[p25,p75] via HB_TRIALS).
    auto bench = [&](int /*iters*/, auto fn) { return hb::bench(fn); };

    Var d("d"), b("b"), n("n");
    auto energy_density = [&](Func y, RVar rd_, Expr time) {
        Expr yi = y(rd_, b, time);
        Expr yip = y(clamp(cast<int>(rd_) + 1, 0, D - 1), b, time);
        Expr g = yip - yi;
        return 0.5f * eps * g * g + 0.25f * (1 - yi * yi) * (1 - yi * yi);
    };

    try {
        // ---------- 1. Inductive dynamics (stencil) + observer ----------
        // fold_k = 3 folds the trajectory to a 3-slice window; fold_k = T pins the
        // full extent (the unfolded ablation), holding fusion/compute placement fixed.
        auto build_ind = [&](int fold_k, const char *nm) -> Func {
            Func y_i(Float(32), std::string("y_i_") + nm), E_i(std::string("E_i_") + nm);
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
            y_i.compute_at(E_i, n).store_root().fold_storage(n, fold_k);
            y_i.update(0).allow_race_conditions().unroll(r.y).vectorize(r.x, 16);
            E_i.bound(b, 0, B).bound(n, 0, T);
            E_i.compile_jit();
            return E_i;
        };
        Func E_i = build_ind(3, "fold");
        Func E_u = build_ind(T, "unfold");

        // ---------- 2. Non-inductive: MATERIALIZE full trajectory, then reduce ----------
        Func y_m(Float(32), "y_m"), E_m("E_m");
        {
            // rd (row) is an RVar, not a plain pure Var, so the stencil's ±1 shifts
            // can't look like a shift of a pure dimension to is_inductive() --
            // matching how y_i above uses r.x for the same role.
            RDom r(0, D, 0, T, "r");
            Expr rd = r.x, rn = r.y;
            y_m(d, b, n) = undef<float>();
            Expr p1 = clamp(rn - 1, 0, T - 1), p2 = clamp(rn - 2, 0, T - 1);
            auto lap = [&](Expr t) {
                return y_m(clamp(rd - 1, 0, D - 1), b, t) - 2.0f * y_m(rd, b, t) +
                       y_m(clamp(rd + 1, 0, D - 1), b, t);
            };
            auto f = [&](Expr t) { Expr c = y_m(rd, b, t); return eps * lap(t) + c - c * c * c; };
            Expr general = y_m(rd, b, p1) + h * (1.5f * f(p1) - 0.5f * f(p2));
            y_m(rd, b, rn) = select(rn == 0, y0(rd, b), general);

            RDom rdE(0, D, "rdE");
            E_m(b, n) = cast<float>(0);
            E_m(b, n) += energy_density(y_m, rdE.x, n);
            y_m.compute_root();  // full D*B*T
            y_m.update(0).reorder(r.x, r.y, b).allow_race_conditions().vectorize(r.x, 16);
            E_m.compute_root();
            E_m.update(0).reorder(rdE.x, b, n);
            E_m.bound(b, 0, B).bound(n, 0, T);
            E_m.compile_jit();
        }

        Buffer<float> ei(B, T), eu(B, T), em(B, T);
        E_i.realize(ei);
        E_u.realize(eu);
        E_m.realize(em);
        hb::Stats s_i = bench(5, [&]() { E_i.realize(ei); });
        hb::Stats s_u = bench(5, [&]() { E_u.realize(eu); });
        hb::Stats s_m = bench(5, [&]() { E_m.realize(em); });

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
        hb::Stats s_boost = bench(5, [&]() {
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
        hb::Stats s_cpp = bench(5, [&]() {
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

        // Sparse (O(D)/step) stencil: once the materialized O(D*B*T) trajectory
        // spills cache, the folded O(D*B*3) inductive form wins on SPEED as well
        // as memory. Both Halide variants checked; Boost/C++ are baselines.
        const double bytes_ind = (double)D * B * 3 * 4;
        const double bytes_non = (double)D * B * T * 4;
        const double thr = (double)D * (double)T * B / 1e6;  // M state-updates
        char note[144];
        snprintf(note, sizeof(note),
                 "SPARSE Allen-Cahn free-energy observer  D=%d B=%d T=%d  |  state fold %.0fx (%.2f -> %.2f MB)",
                 D, B, T, hb::mem_ratio(bytes_non, bytes_ind), hb::mb(bytes_non), hb::mb(bytes_ind));
        hb::print_spec_header("ode_observer_sparse", "host", note);
        hb::print_row("Boost.odeint + observer (3rd-party)", s_boost, thr / (s_boost.min * 1e-3),
                      "Mupd/s", (double)D * 4, err_b, err_b < 1e-3);
        hb::print_row("C++ reference + observer (oracle)", s_cpp, thr / (s_cpp.min * 1e-3),
                      "Mupd/s", 0.0, 0.0, true);
        hb::print_row("Halide non-inductive (materialize)", s_m, thr / (s_m.min * 1e-3),
                      "Mupd/s", bytes_non, err_m, err_m < 1e-3);
        hb::print_row("Halide inductive (fold n -> 3)", s_i, thr / (s_i.min * 1e-3),
                      "Mupd/s", bytes_ind, err_i, err_i < 1e-3, hb::verdict(s_i.min, s_m.min));
        printf("  speedup inductive vs materialize: %.2fx\n", s_m.min / s_i.min);
        bool pass = err_i < 1e-3 && err_m < 1e-3 && err_b < 1e-3;
        printf("  -> %s\n", pass ? "PASS" : "FAIL");
        return pass ? 0 : 1;
    } catch (const Halide::Error &ex) {
        printf("HALIDE ERROR: %s\n", ex.what());
        return 2;
    }
}
