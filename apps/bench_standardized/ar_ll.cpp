// Latent AR(2) + observation-noise Kalman LOG-LIKELIHOOD over a batch of
// independent series -- the LONG-SEQUENCE analogue of kalman_ll_llt.cpp. Unlike
// the local-linear-trend model (a short-T econometric model), a latent AR(p)
// signal observed with noise is a signal-processing filter whose realistic
// sequence length is genuinely large (audio / EEG / seismic / tick data, with
// T from 1e4 up), so the inductive fold-t->2 win lands in a regime that real
// workloads for THIS model actually occupy.
//
// Model (n_states = 2, scalar observation -> only 1/S, NO matrix inverse):
//   latent   s_t = phi1 s_{t-1} + phi2 s_{t-2} + eta_t,  eta_t ~ N(0, q)
//   observed z_t = s_t + r_t,                            r_t   ~ N(0, R)
// State x = [s_t, s_{t-1}].  F = [[phi1, phi2],[1, 0]],  H = [1, 0],
//   Q = diag(q, 0),  R scalar.
// Kalman recursion (posterior covariance P stored; prior P- inline):
//   P-  = F P Fᵀ + Q  ->  Pp00 = phi1² P00 + 2 phi1 phi2 P01 + phi2² P11 + q
//                         Pp01 = phi1 P00 + phi2 P01,   Pp11 = P00
//   S   = Pp00 + R,   K0 = Pp00/S, K1 = Pp01/S
//   x-  = F x = [phi1 x0 + phi2 x1, x0],   innov = z - x-0
//   x   = x- + K innov,   P = (I - K H) P-
//   ll_contrib_t = -0.5 (innov²/S + log S);   LL(b) = Σ_t ll_contrib_t
//
// The five scalar state channels (x0, x1, P00, P01, P11) are PACKED into one
// Func State(b, c, t), inductive in t. LL(b) reduces the whole trajectory over
// t, so the non-inductive form must materialize O(B*T*5) while inductive folds
// t to 2. Same packing/schedule as kalman_ll_llt.cpp; only F, Q differ.
//
// Build: g++ apps/kalman_ll/ar_ll.cpp -O3 -march=native -fopenmp
//   -Iinclude -Lbuild/src -lHalide -lpthread -ldl -o /tmp/kar -std=c++17
//   LD_LIBRARY_PATH=build/src /tmp/kar [B T]

#include "../support/bench_harness.h"
#include "Halide.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int B = argc > 1 ? atoi(argv[1]) : 256;
    int T = argc > 2 ? atoi(argv[2]) : 16384;

    // Stable AR(2): roots inside unit circle (phi1+phi2<1, phi2-phi1<1, |phi2|<1).
    const double phi1 = 0.6, phi2 = 0.3, q = 0.1, Rv = 1.0;
    const int NC = 5;  // packed channels: 0=x0 1=x1 2=P00 3=P01 4=P11

    // Arithmetic-intensity ablation (HB_CHEAP): replace the two per-step divides
    // and the observer log(S) with a fixed-reciprocal multiply, holding the
    // recurrence order, channel packing, memory access pattern, and schedule
    // fixed. This flips the kernel from latency-bound (divide/log critical path)
    // to bandwidth-bound WITHOUT changing the footprint -- so it isolates why
    // folding loses on the real kernel but should win once the arithmetic is
    // cheap. It changes the numeric result, so these runs report UNCHECKED.
    const bool cheap = getenv("HB_CHEAP") != nullptr;
    const double invS = 1.0 / (q + Rv);  // fixed reciprocal used in cheap mode

    Buffer<double> z(B, T);
    srand(7);
    // Box-Muller standard normal from rand().
    auto nrand = [&]() {
        double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
        double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
    };
    // Simulate a latent AR(2) signal observed with additive noise.
    for (int b = 0; b < B; b++) {
        double s1 = 0.0, s2 = 0.0;  // s_{t-1}, s_{t-2}
        for (int t = 0; t < T; t++) {
            double s = phi1 * s1 + phi2 * s2 + std::sqrt(q) * nrand();
            z(b, t) = s + std::sqrt(Rv) * nrand();
            s2 = s1;
            s1 = s;
        }
    }

    auto init_c = [&](Expr c) {
        // x0=0, x1=0, P0 = I  (P00=1, P01=0, P11=1)
        return select(c == 2 || c == 4, Expr(1.0), Expr(0.0));
    };

    auto build = [&](bool inductive, int fold_k = 2) -> Func {
        Var b("b"), c("c"), t("t");
        Func State(Float(64), "State");
        RDom r;

        // Given the five previous-column channel Exprs, produce the new channel c.
        auto step = [&](Expr c, Expr x0, Expr x1, Expr P00, Expr P01, Expr P11, Expr zt) {
            Expr Pp00 = Expr(phi1 * phi1) * P00 + Expr(2.0 * phi1 * phi2) * P01 +
                        Expr(phi2 * phi2) * P11 + Expr(q);
            Expr Pp01 = Expr(phi1) * P00 + Expr(phi2) * P01;
            Expr Pp11 = P00;
            Expr S = Pp00 + Expr(Rv);
            // Cheap mode: fixed-reciprocal multiply instead of a divide (same
            // access pattern, far shorter latency chain).
            Expr K0 = cheap ? Pp00 * Expr(invS) : Pp00 / S;
            Expr K1 = cheap ? Pp01 * Expr(invS) : Pp01 / S;
            Expr xm0 = Expr(phi1) * x0 + Expr(phi2) * x1, xm1 = x0;
            Expr innov = zt - xm0;
            Expr c0 = xm0 + K0 * innov;
            Expr c1 = xm1 + K1 * innov;
            Expr c2 = (Expr(1.0) - K0) * Pp00;
            Expr c3 = (Expr(1.0) - K0) * Pp01;
            Expr c4 = Pp11 - K1 * Pp01;
            return select(c == 0, c0, c == 1, c1, c == 2, c2, c == 3, c3, c4);
        };

        if (inductive) {
            // Pack the 5 channels along an RVar (rc), not the pure Var c: the
            // per-channel self-references read fixed channels of the previous time
            // (State(b,0,t-1), ...), which would otherwise make the classifier treat
            // c as an inductive axis and misread the channel select as a base case.
            // The sole inductive axis is t (base case t<=0, self-reference t-1).
            State(b, c, t) = undef<double>();
            RDom rc(0, NC, "rc");
            Expr x0 = State(b, 0, t - 1), x1 = State(b, 1, t - 1);
            Expr P00 = State(b, 2, t - 1), P01 = State(b, 3, t - 1), P11 = State(b, 4, t - 1);
            State(b, rc, t) = select(t <= 0, init_c(rc),
                                     likely(step(rc, x0, x1, P00, P01, P11, z(b, t))));
        } else {
            State(b, c, t) = init_c(c);
            r = RDom(0, NC, 1, T - 1, "r");  // r.x = channel, r.y = time
            Expr x0 = State(b, 0, r.y - 1), x1 = State(b, 1, r.y - 1);
            Expr P00 = State(b, 2, r.y - 1), P01 = State(b, 3, r.y - 1), P11 = State(b, 4, r.y - 1);
            State(b, r.x, r.y) = step(r.x, x0, x1, P00, P01, P11, z(b, r.y));
        }

        // Consumer: per-series log-likelihood, a reduction over t.
        Func LL("LL");
        RDom rl(1, T - 1, "rl");
        Expr P00 = State(b, 2, rl - 1), P01 = State(b, 3, rl - 1), P11 = State(b, 4, rl - 1);
        Expr Pp00 = Expr(phi1 * phi1) * P00 + Expr(2.0 * phi1 * phi2) * P01 +
                    Expr(phi2 * phi2) * P11 + Expr(q);
        Expr S = Pp00 + Expr(Rv);
        Expr xm0 = Expr(phi1) * State(b, 0, rl - 1) + Expr(phi2) * State(b, 1, rl - 1);
        Expr innov = z(b, rl) - xm0;
        LL(b) = Expr(0.0);
        // Cheap mode drops the observer divide and log (see HB_CHEAP note above).
        LL(b) += cheap ? Expr(-0.5) * (innov * innov * Expr(invS)) : Expr(-0.5) * (innov * innov / S + log(S));

        const int V = 8;
        Var bo("bo"), bi("bi");
        LL.bound(b, 0, B).split(b, bo, bi, V).vectorize(bi).parallel(bo);
        LL.update(0).split(b, bo, bi, V).reorder(bi, rl, bo).vectorize(bi).parallel(bo);
        if (inductive) {
            State.reorder_storage(b, c, t).compute_at(LL, rl).store_at(LL, bo).fold_storage(t, fold_k);
        } else {
            State.reorder_storage(b, c, t).compute_at(LL, bo).vectorize(b, V);
            State.update(0).reorder(b, r.x, r.y).vectorize(b, V);
        }
        return LL;
    };

    try {
        Func li = build(true, 2), lu = build(true, T), ln = build(false);
        li.compile_jit();
        lu.compile_jit();
        ln.compile_jit();
        Buffer<double> ri(B), ru(B), rn(B);
        li.realize(ri);
        lu.realize(ru);
        ln.realize(rn);

        hb::Stats sn = hb::bench([&] { ln.realize(rn); });
        hb::Stats su = hb::bench([&] { lu.realize(ru); });
        hb::Stats si = hb::bench([&] { li.realize(ri); });

        // Scalar C++ reference.
        std::vector<double> cll((size_t)B);
        {
            const double *zp = z.data();
#pragma omp parallel for schedule(static)
            for (int b = 0; b < B; b++) {
                double x0 = 0, x1 = 0, P00 = 1, P01 = 0, P11 = 1, ll = 0;
                for (int t = 1; t < T; t++) {
                    double Pp00 = phi1 * phi1 * P00 + 2 * phi1 * phi2 * P01 + phi2 * phi2 * P11 + q;
                    double Pp01 = phi1 * P00 + phi2 * P01;
                    double Pp11 = P00;
                    double S = Pp00 + Rv, K0 = Pp00 / S, K1 = Pp01 / S;
                    double xm0 = phi1 * x0 + phi2 * x1, xm1 = x0;
                    double innov = zp[(size_t)b + (size_t)t * B] - xm0;
                    ll += -0.5 * (innov * innov / S + std::log(S));
                    x0 = xm0 + K0 * innov;
                    x1 = xm1 + K1 * innov;
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
            double a = ri(b), au = ru(b), cc = rn(b), g = cll[b];
            if (std::isnan(a) || std::isnan(au) || std::isnan(cc)) bad = true;
            err = std::max({err, std::abs(a - g), std::abs(au - g), std::abs(cc - g)});
        }
        // Analytic state footprint: inductive folds t -> 2 slices (O(B*NC*2));
        // the unfolded ablation and non-inductive both materialize O(B*NC*T).
        const double bytes_ind = (double)NC * B * 2 * 8;
        const double bytes_unf = (double)NC * B * T * 8;
        const double bytes_non = (double)NC * B * T * 8;
        bool ok = !bad && err < 1e-5;
        // Cheap-arithmetic runs change the numeric result on purpose, so they are
        // an arithmetic-intensity probe, not a correctness comparison: report
        // UNCHECKED (err < 0) rather than a meaningless FAIL against the exact ref.
        double rep_err = cheap ? -1.0 : err;
        char note[200];
        snprintf(note, sizeof(note),
                 "Latent AR(2)+obs-noise log-likelihood  B=%d T=%d  |  state fold %.0fx (%.2f -> %.2f MB)  |  unfolded fp/LLC=%.3f%s",
                 B, T, hb::mem_ratio(bytes_non, bytes_ind), hb::mb(bytes_non), hb::mb(bytes_ind),
                 hb::footprint_over_llc(bytes_unf), cheap ? "  [HB_CHEAP: latency->bandwidth]" : "");
        hb::print_spec_header("kalman_ar (ar_ll)", "host", note);
        // Throughput = the actual work: B sequences x T Kalman likelihood-update
        // steps (the old B/1e6 ignored T and rounded to 0.0 Mseq/s).
        double msteps = (double)B * T / 1e6;
        // Honest verdict from measured time (2% noise band); the inductive win here
        // is primarily the O(B*NC*T) -> O(B*NC*2) memory fold shown in state_MB,
        // and speed is config-dependent, so don't hard-code "win". The trailing
        // fp/LLC (unfolded footprint / last-level cache) is the roofline x-axis:
        // recurrence-length and batch-size sweeps collapse onto it.
        hb::print_row("non-inductive (materialize)", sn, msteps / (sn.min * 1e-3), "Mstep/s",
                      bytes_non, rep_err, ok, "", bytes_unf);
        hb::print_row("inductive UNFOLDED (fold t -> T)", su, msteps / (su.min * 1e-3), "Mstep/s",
                      bytes_unf, rep_err, ok, hb::verdict(su.min, sn.min), bytes_unf);
        hb::print_row("inductive FOLDED (fold t -> 2)", si, msteps / (si.min * 1e-3), "Mstep/s",
                      bytes_ind, rep_err, ok, hb::verdict(si.min, su.min), bytes_unf);

        auto dump = [](const char *p, const double *d, size_t n) {
            FILE *f = fopen(p, "wb"); if (f) { fwrite(d, sizeof(double), n, f); fclose(f); } };
        dump("apps/kalman_ll/z_ar.bin", z.data(), (size_t)B * T);
        std::vector<double> llflat(ri.data(), ri.data() + B);
        dump("apps/kalman_ll/ll_ar.bin", llflat.data(), B);
        FILE *fp = fopen("apps/kalman_ll/params_ar.txt", "w");
        if (fp) {
            fprintf(fp, "%d %d %.6f %.6f %.9f %.9f %.9f %.9f\n",
                    B, T, si.min, sn.min, phi1, phi2, q, Rv);
            fclose(fp);
        }
        // Cheap mode is an arithmetic-intensity probe (numerics change on
        // purpose), so success is just "no NaN" -- not the exact-ref check, which
        // would spuriously fail the run. Non-cheap keeps the strict correctness gate.
        return (cheap ? !bad : (!bad && err < 1e-5)) ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
