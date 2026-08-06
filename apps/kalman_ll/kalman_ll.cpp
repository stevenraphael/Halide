// 1-D Kalman filter LOG-LIKELIHOOD over a batch of independent series -- the
// real simdkalman workload (parameter fitting / model selection): run the FULL
// filter (with the covariance/Riccati recursion) and reduce each series' whole
// trajectory to ONE scalar log-likelihood.
//
// Local-level model per series b:  x_t = F x_{t-1} + w,  z_t = H x_t + v.
// Full filter, storing the POSTERIOR covariance P; the PRIOR P- is an inline temp:
//   P-_t = F^2 P_{t-1} + Q         (predict)
//   S_t  = H^2 P-_t  + R           (innovation variance)
//   K_t  = H P-_t / S_t            (gain)
//   P_t  = (1 - K_t H) P-_t        (update, posterior -- the stored recurrence)
//   xp_t = F x_{t-1};  innov = z_t - H xp_t;  x_t = xp_t + K_t innov
//   ll_contrib_t = -0.5 (innov^2 / S_t + log S_t)     (matches simdkalman)
//   LL(b) = sum_t ll_contrib_t
//
// The consumer LL(b) REDUCES the (x,P) trajectory over t to one scalar per series.
// That reduce-over-the-fold-axis consumer is why the non-inductive form cannot
// fold: its t-loop is an RVar accumulation, so no producer can be sliced into it
// (a manual circular buffer works only when the consumer reads a per-t slice, as
// in kalman_ss). Inductive folds x,P to a 2-slice window fused into LL -> O(B)
// memory; non-inductive must materialize the O(B*T) trajectory (like simdkalman).
//
// Build: g++ apps/kalman_ll/kalman_ll.cpp -O3 -march=native -fopenmp -Iinclude
//        -Lbuild/src -lHalide -lpthread -ldl -o /tmp/kll -std=c++17
//        LD_LIBRARY_PATH=build/src /tmp/kll [B T]

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int B = argc > 1 ? atoi(argv[1]) : 4096;  // number of independent series
    int T = argc > 2 ? atoi(argv[2]) : 512;   // series length (timesteps)

    const double F = 1.0, H = 1.0;    // local-level model
    const double x0 = 0.0, P0 = 1.0;  // initial posterior state/cov

    // SHARED process/observation noise (one model, B series) -- matches
    // simdkalman, whose KalmanFilter matrices are shared across series (its
    // canonical use: score/fit many series against a single model). P is then
    // the same deterministic sequence for every series; the per-series x
    // trajectory (B*T) is the large intermediate the log-likelihood folds away.
    const double Qv = 0.05, Rv = 1.0;
    Buffer<double> z(B, T);
    srand(7);
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++)
            z(b, t) = (rand() % 2000) / 1000.0 - 1.0;

    auto build = [&](bool inductive) -> Func {
        Var b("b"), t("t");
        Func P(Float(64), "P"), x(Float(64), "x");
        Expr Fh = Expr(F), Hh = Expr(H), Q = Expr(Qv), R = Expr(Rv);

        if (inductive) {
            // Posterior P and state x as inductive scans over t; P- inline.
            Expr Pp_i = Fh * Fh * P(b, t - 1) + Q;
            Expr S_i = Hh * Hh * Pp_i + R;
            Expr K_i = Hh * Pp_i / S_i;
            P(b, t) = select(t <= 0, Expr(P0),
                             likely((Expr(1.0) - K_i * Hh) * Pp_i));
            Expr xp_i = Fh * x(b, t - 1);
            x(b, t) = select(t <= 0, Expr(x0),
                             likely(xp_i + K_i * (z(b, t) - Hh * xp_i)));
        } else {
            // Idiomatic non-inductive: RDom-t scans, materialized (compute_root).
            P(b, t) = Expr(P0);
            x(b, t) = Expr(x0);
            RDom rt(1, T - 1);
            Expr Pp = Fh * Fh * P(b, rt - 1) + Q;
            Expr S = Hh * Hh * Pp + R;
            Expr K = Hh * Pp / S;
            P(b, rt) = (Expr(1.0) - K * Hh) * Pp;
            Expr xp = Fh * x(b, rt - 1);
            x(b, rt) = xp + K * (z(b, rt) - Hh * xp);
        }

        // Consumer: per-series log-likelihood -- a REDUCTION over t. Uses the
        // PRIOR P- and xp, recomputed inline from the previous posteriors.
        Func LL("LL");
        RDom rl(1, T - 1, "rl");
        Expr Pp = Fh * Fh * P(b, rl - 1) + Q;
        Expr S = Hh * Hh * Pp + R;
        Expr xp = Fh * x(b, rl - 1);
        Expr innov = z(b, rl) - Hh * xp;
        LL(b) = Expr(0.0);
        LL(b) += Expr(-0.5) * (innov * innov / S + log(S));

        const int V = 8;
        Var bo("bo"), bi("bi");
        LL.bound(b, 0, B).split(b, bo, bi, V).vectorize(bi).parallel(bo);
        LL.update(0).split(b, bo, bi, V).reorder(bi, rl, bo).vectorize(bi).parallel(bo);
        // Store x,P with batch innermost so vectorizing over b is contiguous.
        if (inductive) {
            x.reorder_storage(b, t).compute_at(LL, rl).store_at(LL, bo).fold_storage(t, 2).vectorize(b, V);
            P.reorder_storage(b, t).compute_at(LL, rl).store_at(LL, bo).fold_storage(t, 2).vectorize(b, V);
        } else {
            x.reorder_storage(b, t).compute_at(LL, bo).vectorize(b, V);
            x.update(0).vectorize(b, V);
            P.reorder_storage(b, t).compute_at(LL, bo).vectorize(b, V);
            P.update(0).vectorize(b, V);
        }
        return LL;
    };

    try {
        Func li = build(true), ln = build(false);
        li.compile_jit();
        ln.compile_jit();
        Buffer<double> ri(B), rn(B);
        li.realize(ri);
        ln.realize(rn);

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
        double ti = bench(li, ri), tn = bench(ln, rn);

        // C++ streaming reference (scalar per-series filter + loglik).
        std::vector<double> cll((size_t)B);
        {
            const double *zp = z.data();
#pragma omp parallel for schedule(static)
            for (int b = 0; b < B; b++) {
                double xpost = x0, Ppost = P0, ll = 0.0;
                for (int t = 1; t < T; t++) {
                    double Pp = F * F * Ppost + Qv;
                    double S = H * H * Pp + Rv;
                    double K = H * Pp / S;
                    double xp = F * xpost;
                    double innov = zp[(size_t)b + (size_t)t * B] - H * xp;
                    ll += -0.5 * (innov * innov / S + std::log(S));
                    xpost = xp + K * innov;
                    Ppost = (1.0 - K * H) * Pp;
                }
                cll[b] = ll;
            }
        }

        double err = 0;
        bool bad = false;
        for (int b = 0; b < B; b++) {
            double a = ri(b), c = rn(b), g = cll[b];
            if (std::isnan(a) || std::isnan(c)) bad = true;
            err = std::max({err, std::abs(a - g), std::abs(c - g)});
        }
        double traj = (double)2 * B * T * 8 / (1024.0 * 1024.0);
        printf("Kalman log-likelihood  B=%d T=%d  ((x,P) trajectory %.0f MB)\n", B, T, traj);
        printf("  inductive Halide (fold x,P -> 2):  %8.3f ms\n", ti);
        printf("  non-inductive Halide (materialize):%8.3f ms\n", tn);
        printf("  max abs err %.3g%s -> %s\n", err, bad ? " (NaN)" : "",
               (!bad && err < 1e-6) ? "PASS" : "FAIL");

        // Dump inputs + Halide log-likelihood for the simdkalman third-party bench.
        auto dump = [](const char *p, const double *d, size_t n) {
            FILE *f = fopen(p, "wb"); if (f) { fwrite(d, sizeof(double), n, f); fclose(f); } };
        dump("apps/kalman_ll/z.bin", z.data(), (size_t)B * T);
        std::vector<double> llflat(ri.data(), ri.data() + B);
        dump("apps/kalman_ll/ll.bin", llflat.data(), B);
        FILE *fp = fopen("apps/kalman_ll/params.txt", "w");
        if (fp) {
            fprintf(fp, "%d %d %.6f %.6f %.9f %.9f %.9f %.9f\n",
                    B, T, ti, tn, F, H, Qv, Rv);
            fclose(fp);
        }
        return (!bad && err < 1e-6) ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
