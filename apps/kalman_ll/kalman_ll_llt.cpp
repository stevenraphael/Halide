// Local-linear-trend Kalman filter LOG-LIKELIHOOD over a batch of independent
// series -- the realistic (n_states = 2) version of kalman_ll.cpp. This is a
// genuinely-used simdkalman model, not a degenerate 1-state one: the covariance
// P is now a full 2x2 symmetric matrix that evolves per step (not a scalar), so
// the per-step work and the folded state are both real.
//
// State per series b: x = [level, trend]. Model
//   F = [[1,1],[0,1]]   (level += trend)          H = [1, 0]   (observe level)
//   Q = diag(q0, q1)    (process noise)           R  = scalar  (obs noise)
// Full filter (posterior covariance P stored; prior P- inline):
//   P-  = F P Fᵀ + Q
//   S   = H P- Hᵀ + R,   K = P- Hᵀ / S,   P = (I - K H) P-
//   x-  = F x,  innov = z - H x-,  x = x- + K innov
//   ll_contrib_t = -0.5 (innov²/S + log S);   LL(b) = Σ_t ll_contrib_t
//
// The five scalar state channels (level, trend, P00, P01, P11) are PACKED into
// one Func State(b, c, t), inductive in t (the c-index self-calls are constants,
// so detection treats c as an ordinary pure/parallel dimension). LL(b) reduces
// the whole trajectory over t, so -- exactly as in the 1-D case -- the
// non-inductive form must materialize O(B*T*5) while inductive folds t to 2.
//
// Build: g++ apps/kalman_ll/kalman_ll_llt.cpp -O3 -march=native -fopenmp
//   -Iinclude -Lbuild/src -lHalide -lpthread -ldl -o /tmp/kllt -std=c++17
//   LD_LIBRARY_PATH=build/src /tmp/kllt [B T]

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int B = argc > 1 ? atoi(argv[1]) : 256;
    int T = argc > 2 ? atoi(argv[2]) : 16384;

    // Local-linear-trend model parameters (shared across series, per simdkalman).
    const double q0 = 0.01, q1 = 0.0001, Rv = 1.0;  // process/obs noise
    const int NC = 5;                               // packed channels: 0=level 1=trend 2=P00 3=P01 4=P11

    Buffer<double> z(B, T);
    srand(7);
    // A drifting-level series so the trend state is exercised.
    for (int b = 0; b < B; b++) {
        double lvl = 0.0, tr = (rand() % 200) / 1000.0 - 0.1;
        for (int t = 0; t < T; t++) {
            lvl += tr;
            z(b, t) = lvl + ((rand() % 2000) / 1000.0 - 1.0);
        }
    }

    // Packed init and one-step update, shared by both builds. Reads the five
    // previous-column channels via prev(c') and returns the new channel `newc`.
    auto init_c = [&](Expr c) {
        // level=0, trend=0, P0 = I  (P00=1, P01=0, P11=1)
        return select(c == 2 || c == 4, Expr(1.0), Expr(0.0));
    };

    auto build = [&](bool inductive) -> Func {
        Var b("b"), c("c"), t("t");
        Func State(Float(64), "State");
        RDom r;  // non-inductive channel/time reduction domain (assigned below)

        // Given the five previous-column channel Exprs, produce the new channel
        // selected by `c`.
        auto step = [&](Expr c, Expr lp, Expr bp, Expr P00, Expr P01, Expr P11, Expr zt) {
            Expr Pp00 = P00 + Expr(2.0) * P01 + P11 + Expr(q0);
            Expr Pp01 = P01 + P11;
            Expr Pp11 = P11 + Expr(q1);
            Expr S = Pp00 + Expr(Rv);
            Expr K0 = Pp00 / S, K1 = Pp01 / S;
            Expr xl = lp + bp, xb = bp;
            Expr innov = zt - xl;
            Expr c0 = xl + K0 * innov;
            Expr c1 = xb + K1 * innov;
            Expr c2 = (Expr(1.0) - K0) * Pp00;
            Expr c3 = (Expr(1.0) - K0) * Pp01;
            Expr c4 = Pp11 - K1 * Pp01;
            return select(c == 0, c0, c == 1, c1, c == 2, c2, c == 3, c3, c4);
        };

        if (inductive) {
            Expr lp = State(b, 0, t - 1), bp = State(b, 1, t - 1);
            Expr P00 = State(b, 2, t - 1), P01 = State(b, 3, t - 1), P11 = State(b, 4, t - 1);
            State(b, c, t) = select(t <= 0, init_c(c),
                                    likely(step(c, lp, bp, P00, P01, P11, z(b, t))));
        } else {
            // Idiomatic non-inductive: channel AND time are RVars of one 2-D
            // reduction domain (time outermost). Because every recursing
            // position is an RVar, this is an ordinary scan -- NOT detected
            // inductive -- and it materializes the whole O(B*T*NC) trajectory.
            State(b, c, t) = init_c(c);
            r = RDom(0, NC, 1, T - 1, "r");  // r.x = channel, r.y = time
            Expr lp = State(b, 0, r.y - 1), bp = State(b, 1, r.y - 1);
            Expr P00 = State(b, 2, r.y - 1), P01 = State(b, 3, r.y - 1), P11 = State(b, 4, r.y - 1);
            State(b, r.x, r.y) = step(r.x, lp, bp, P00, P01, P11, z(b, r.y));
        }

        // Consumer: per-series log-likelihood, a reduction over t.
        Func LL("LL");
        RDom rl(1, T - 1, "rl");
        Expr P00 = State(b, 2, rl - 1), P01 = State(b, 3, rl - 1), P11 = State(b, 4, rl - 1);
        Expr Pp00 = P00 + Expr(2.0) * P01 + P11 + Expr(q0);
        Expr S = Pp00 + Expr(Rv);
        Expr xl = State(b, 0, rl - 1) + State(b, 1, rl - 1);
        Expr innov = z(b, rl) - xl;
        LL(b) = Expr(0.0);
        LL(b) += Expr(-0.5) * (innov * innov / S + log(S));

        const int V = 8;
        Var bo("bo"), bi("bi");
        LL.bound(b, 0, B).split(b, bo, bi, V).vectorize(bi).parallel(bo);
        LL.update(0).split(b, bo, bi, V).reorder(bi, rl, bo).vectorize(bi).parallel(bo);
        if (inductive) {
            State.reorder_storage(b, c, t).compute_at(LL, rl).store_at(LL, bo).fold_storage(t, 2).vectorize(b, V);
        } else {
            State.reorder_storage(b, c, t).compute_at(LL, bo).vectorize(b, V);
            State.update(0).reorder(b, r.x, r.y).vectorize(b, V);
        }
        return LL;
    };

    // -------- extra-dimension trick: fold the NON-INDUCTIVE form too --------
    // Internalize the log-likelihood as a 6th state channel (acc). Then the
    // consumer is an ENDPOINT read State(b, acc, T-1), NOT a reduction over t.
    // With nothing reducing over t, the scan folds -- here MANUALLY, via a
    // circular time buffer of width W=2 (the state func's time dim has extent 2,
    // indexed rt%W). This shows folding is a property of the recurrence +
    // consumer, achievable non-inductively with restructuring: expressiveness,
    // not a capability unique to inductive funcs.
    auto build_folded = [&]() -> Func {
        Var b("b"), c("c");
        const int NCA = 6, W = 2;  // channels 0..4 as before, 5 = ll accumulator
        Func S(Float(64), "Sfold");
        RDom r(0, NCA, 1, T - 1, "rf");  // r.x = channel, r.y = time
        Expr ps = (r.y - 1) % W;         // previous slot
        Expr lp = S(b, 0, ps), bp = S(b, 1, ps);
        Expr P00 = S(b, 2, ps), P01 = S(b, 3, ps), P11 = S(b, 4, ps), acc = S(b, 5, ps);
        Expr Pp00 = P00 + Expr(2.0) * P01 + P11 + Expr(q0);
        Expr Pp01 = P01 + P11, Pp11 = P11 + Expr(q1);
        Expr Sv = Pp00 + Expr(Rv), K0 = Pp00 / Sv, K1 = Pp01 / Sv;
        Expr xl = lp + bp, innov = z(b, r.y) - xl;
        Expr contrib = Expr(-0.5) * (innov * innov / Sv + log(Sv));
        Expr nv = select(r.x == 0, xl + K0 * innov,
                         r.x == 1, bp + K1 * innov,
                         r.x == 2, (Expr(1.0) - K0) * Pp00,
                         r.x == 3, (Expr(1.0) - K0) * Pp01,
                         r.x == 4, Pp11 - K1 * Pp01,
                         acc + contrib);
        S(b, c, Var("w")) = select(c == 2 || c == 4, Expr(1.0), Expr(0.0));  // acc(w=0)=0
        S(b, r.x, r.y % W) = nv;

        Func LLf("LLf");
        LLf(b) = S(b, 5, (T - 1) % W);  // endpoint read: no reduction over t

        const int V = 8;
        Var bo("bo"), bi("bi");
        LLf.bound(b, 0, B).split(b, bo, bi, V).vectorize(bi).parallel(bo);
        // Fold: time dim has extent W=2 by construction -> O(B*NCA*2) memory.
        S.reorder_storage(b, c, Var("w")).compute_at(LLf, bo).vectorize(b, V);
        S.update(0).reorder(b, r.x, r.y).vectorize(b, V);
        return LLf;
    };

    try {
        Func li = build(true), ln = build(false), lf = build_folded();
        li.compile_jit();
        ln.compile_jit();
        lf.compile_jit();
        Buffer<double> ri(B), rn(B), rf(B);
        li.realize(ri);
        ln.realize(rn);
        lf.realize(rf);

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
        double ti = bench(li, ri), tn = bench(ln, rn), tf = bench(lf, rf);

        // Scalar C++ reference.
        std::vector<double> cll((size_t)B);
        {
            const double *zp = z.data();
#pragma omp parallel for schedule(static)
            for (int b = 0; b < B; b++) {
                double l = 0, tr = 0, P00 = 1, P01 = 0, P11 = 1, ll = 0;
                for (int t = 1; t < T; t++) {
                    double Pp00 = P00 + 2 * P01 + P11 + q0;
                    double Pp01 = P01 + P11;
                    double Pp11 = P11 + q1;
                    double S = Pp00 + Rv, K0 = Pp00 / S, K1 = Pp01 / S;
                    double xl = l + tr, innov = zp[(size_t)b + (size_t)t * B] - xl;
                    ll += -0.5 * (innov * innov / S + std::log(S));
                    l = xl + K0 * innov;
                    tr = tr + K1 * innov;
                    P00 = (1 - K0) * Pp00;
                    P01 = (1 - K0) * Pp01;
                    P11 = Pp11 - K1 * Pp01;
                }
                cll[b] = ll;
            }
        }

        double err = 0;
        bool bad = false;
        for (int b = 0; b < B; b++) {
            double a = ri(b), cc = rn(b), ff = rf(b), g = cll[b];
            if (std::isnan(a) || std::isnan(cc) || std::isnan(ff)) bad = true;
            err = std::max({err, std::abs(a - g), std::abs(cc - g), std::abs(ff - g)});
        }
        double traj = (double)NC * B * T * 8 / (1024.0 * 1024.0);
        printf("Kalman local-linear-trend log-likelihood  B=%d T=%d  (state traj %.0f MB)\n", B, T, traj);
        printf("  inductive Halide (fold t->2):           %8.3f ms\n", ti);
        printf("  non-inductive Halide (materialize):     %8.3f ms\n", tn);
        printf("  non-inductive + acc channel (manual fold):%7.3f ms\n", tf);
        printf("  max abs err %.3g%s -> %s\n", err, bad ? " (NaN)" : "",
               (!bad && err < 1e-5) ? "PASS" : "FAIL");

        auto dump = [](const char *p, const double *d, size_t n) {
            FILE *f = fopen(p, "wb"); if (f) { fwrite(d, sizeof(double), n, f); fclose(f); } };
        dump("apps/kalman_ll/z_llt.bin", z.data(), (size_t)B * T);
        std::vector<double> llflat(ri.data(), ri.data() + B);
        dump("apps/kalman_ll/ll_llt.bin", llflat.data(), B);
        FILE *fp = fopen("apps/kalman_ll/params_llt.txt", "w");
        if (fp) {
            fprintf(fp, "%d %d %.6f %.6f %.9f %.9f %.9f\n", B, T, ti, tn, q0, q1, Rv);
            fclose(fp);
        }
        return (!bad && err < 1e-5) ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
