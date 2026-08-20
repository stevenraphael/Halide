// Chebyshev iteration preconditioned by a ONE-LEVEL ADDITIVE SCHWARZ method,
// for the 1-D Poisson system A x = b, A = tridiag(-1, 2, -1).
//
// Preconditioned three-term recurrence (x_{-1}=x_0=0):
//   r_k = b - A x_k
//   z_k = M^{-1} r_k            (additive Schwarz: sum of overlapping local solves)
//   x_{k+1} = (1+w_k) x_k - w_k x_{k-1} + a_k z_k
//
// TWO SLIDING WINDOWS:
//  (1) TEMPORAL  -- the iterate X(t,k) needs columns k-1 and k-2, so its storage
//      folds over the iteration axis k (fold to 3). This is the big O(n*M) win.
//  (2) SPATIAL   -- the additive-Schwarz block corrections Y(l,i) are consumed by
//      z(t) in spatial order (covering block i = t/st - d), each output DOF
//      gathering from ncov = ceil(bs/st) overlapping blocks, so Y folds over the
//      block axis i (fold to ncov+1).
//
// The block inverse Binv (bs x bs) is precomputed on the HOST and passed in --
// forming a matrix inverse is a sequential pivoted algorithm, not Halide's job;
// real Schwarz codes factor the local solves once at setup and only APPLY them.
//
// Build: g++ apps/chebyshev_schwarz/chebyshev_schwarz.cpp -O3 -march=native
//   -fopenmp -Ibuild/include -Lbuild/src -lHalide -lpthread -ldl -o /tmp/cs
//   -std=c++17 ;  LD_LIBRARY_PATH=build/src /tmp/cs [n bs ov M]

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    const int n  = argc > 1 ? atoi(argv[1]) : 1024;
    const int bs = argc > 2 ? atoi(argv[2]) : 32;
    const int ov = argc > 3 ? atoi(argv[3]) : 24;
    const int M  = argc > 4 ? atoi(argv[4]) : 100;
    const int st = bs - ov;
    const int nb = (n - ov + st - 1) / st;   // #blocks covering the domain
    const int ncov = (bs + st - 1) / st;     // #blocks covering each DOF (window 2)

    // ---- host setup: block inverse of bs x bs tridiag(-1,2,-1) (Gauss-Jordan) ----
    std::vector<double> Binvh((size_t)bs * bs, 0.0);
    {
        int m = bs; std::vector<double> Aug((size_t)m * 2 * m, 0.0);
        for (int i = 0; i < m; i++) {
            Aug[i * 2 * m + i] = 2;
            if (i > 0) Aug[i * 2 * m + i - 1] = -1;
            if (i + 1 < m) Aug[i * 2 * m + i + 1] = -1;
            Aug[i * 2 * m + m + i] = 1;
        }
        for (int c = 0; c < m; c++) {
            double piv = Aug[c * 2 * m + c];
            for (int j = 0; j < 2 * m; j++) Aug[c * 2 * m + j] /= piv;
            for (int r = 0; r < m; r++) if (r != c) {
                double f = Aug[r * 2 * m + c];
                for (int j = 0; j < 2 * m; j++) Aug[r * 2 * m + j] -= f * Aug[c * 2 * m + j];
            }
        }
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++)
            Binvh[(size_t)i * m + j] = Aug[i * 2 * m + m + j];
    }

    // ---- host reference apply-ops (also used for spectral bounds + validation) ----
    auto applyA = [&](const std::vector<double> &x, std::vector<double> &y) {
        for (int i = 0; i < n; i++) {
            double v = 2 * x[i];
            if (i > 0) v -= x[i - 1];
            if (i + 1 < n) v -= x[i + 1];
            y[i] = v;
        }
    };
    auto applyMinv = [&](const std::vector<double> &r, std::vector<double> &z) {
        std::fill(z.begin(), z.end(), 0.0);
        for (int blk = 0; blk < nb; blk++) {
            int s = blk * st;
            for (int l = 0; l < bs && s + l < n; l++) {
                double acc = 0;
                for (int lp = 0; lp < bs && s + lp < n; lp++)
                    acc += Binvh[(size_t)l * bs + lp] * r[s + lp];
                z[s + l] += acc;
            }
        }
    };

    // x_exact, b = A x_exact
    std::vector<double> xe(n), bh(n), tmp(n);
    for (int i = 0; i < n; i++) xe[i] = std::sin(0.01 * i) + 0.3 * std::cos(0.03 * i);
    applyA(xe, bh);

    // spectral bounds of B = Minv*A via power iteration (and shifted for lmin)
    auto applyB = [&](const std::vector<double> &v, std::vector<double> &Bv) {
        std::vector<double> t2(n); applyA(v, t2); applyMinv(t2, Bv);
    };
    auto nrm = [&](const std::vector<double> &a) { double s = 0; for (double x : a) s += x * x; return std::sqrt(s); };
    auto power = [&](bool shifted, double shift) {
        std::vector<double> v(n), Bv(n);
        for (int i = 0; i < n; i++) v[i] = 1.0 + 0.001 * i;
        double lam = 0;
        for (int it = 0; it < 300; it++) {
            applyB(v, Bv);
            if (shifted) for (int i = 0; i < n; i++) Bv[i] = shift * v[i] - Bv[i];
            double nn = nrm(Bv); if (nn == 0) break;
            for (int i = 0; i < n; i++) v[i] = Bv[i] / nn;
            lam = nn;
        }
        return lam;
    };
    double lmax = power(false, 0);
    double lmin = lmax - power(true, lmax);

    // Chebyshev coefficients on [lmin, lmax]
    std::vector<double> alh(M), omh(M);
    { double d = 0.5 * (lmax + lmin), c = 0.5 * (lmax - lmin), ap = 0;
      for (int k = 0; k < M; k++) {
          if (k == 0) { alh[k] = 1.0 / d; omh[k] = 0; }
          else { double beta = (c * ap * 0.5) * (c * ap * 0.5); alh[k] = 1.0 / (d - beta / ap); omh[k] = alh[k] * beta / ap; }
          ap = alh[k];
      } }
    printf("Chebyshev + 1-level additive Schwarz   n=%d bs=%d ov=%d nb=%d ncov=%d M=%d\n",
           n, bs, ov, nb, ncov, M);
    printf("  M^-1A spectrum lmin=%.4g lmax=%.4g cond=%.3g\n", lmin, lmax, lmax / lmin);

    Buffer<double> b(bh.data(), n), Binv(Binvh.data(), bs, bs);
    Buffer<double> alpha(alh.data(), M), omega(omh.data(), M);

    // Host-precomputed helpers so the Schwarz apply can be INLINED into X's
    // self-accumulating update using only clamp() (no select around self-refs):
    //  Wlo/Whi : Dirichlet boundary weights for the tridiag stencil.
    //  Cf(t,d,lp): coefficient Binv(loc,lp) for the d-th block covering t, or 0
    //              if that (block,local) pair is out of range. Encodes the whole
    //              additive-Schwarz overlap structure as a per-position weight.
    Buffer<double> Wlo(n), Whi(n);
    for (int tt = 0; tt < n; tt++) { Wlo(tt) = tt > 0 ? 1.0 : 0.0; Whi(tt) = tt < n - 1 ? 1.0 : 0.0; }
    Buffer<double> Cf(n, ncov, bs);
    for (int tt = 0; tt < n; tt++)
        for (int dd = 0; dd < ncov; dd++)
            for (int lpp = 0; lpp < bs; lpp++) {
                int loc = (tt % st) + dd * st;
                int blk = tt / st - dd;
                int gt = blk * st + lpp;
                bool ok = blk >= 0 && blk < nb && loc >= 0 && loc < bs && gt >= 0 && gt < n;
                Cf(tt, dd, lpp) = ok ? Binvh[(size_t)loc * bs + lpp] : 0.0;
            }

    // ---------------------------- Halide inductive ----------------------------
    try {
    Var t("t"), k("k"), l("l"), i("i");

    Func X(Float(64), "X");
    X(t, k) = cast<double>(0);                       // base: x_0 = 0
    Expr km1 = max(0, k - 1), km2 = max(0, k - 2);

    // residual r = b - A x_{k-1}  (tridiag stencil; zero outside [0,n))
    Func r("r");
    Expr xtm = select(t > 0, X(clamp(t - 1, 0, n - 1), km1), cast<double>(0));
    Expr xtp = select(t < n - 1, X(clamp(t + 1, 0, n - 1), km1), cast<double>(0));
    r(t, k) = b(t) - (cast<double>(2) * X(t, km1) - xtm - xtp);

    // block corrections Y(l,i) = Binv * (r restricted to block i)   [window 2 data]
    Func Y("Y");
    RDom lp(0, bs, "lp");
    Expr gt = i * st + lp;                           // global index of local lp in block i
    Y(l, i, k) = sum(Binv(l, lp) * select(gt < n, r(clamp(gt, 0, n - 1), k), cast<double>(0)));

    // z(t) = sum over the ncov blocks covering t of Y(local, block)
    Func z("z");
    RDom d(0, ncov, "d");
    Expr blk = t / st - d;                           // covering block indices (in-order in t)
    Expr loc = t - blk * st;                         // local position within that block
    Expr valid = blk >= 0 && blk < nb && loc >= 0 && loc < bs;
    if (getenv("TRIVIAL_Z")) z(t, k) = r(t, k);      // bisect: skip Schwarz reductions
    else
    z(t, k) = sum(select(valid, Y(clamp(loc, 0, bs - 1), clamp(blk, 0, nb - 1), k), cast<double>(0)));

    // inductive three-term update; self-refs X(t,km1),X(t,km2) sit in one select
    X(t, k) = select(k <= 0, cast<double>(0),
                     likely((cast<double>(1) + omega(km1)) * X(t, km1)
                            - omega(km1) * X(t, km2)
                            + alpha(km1) * z(t, k)));

    // endpoint-extract consumer drives the k-sweep and lets X fold
    Func out("out");
    RDom rk(0, M + 1, "rk");
    out(t) = cast<double>(0);
    out(t) += select(rk == M, X(t, rk), cast<double>(0));
    out.update(0).reorder(t, rk);

        // schedule: correctness-first, everything materialized (no fold yet).
        r.compute_root();
        Y.compute_root();
        z.compute_root();
        X.compute_root();

        Buffer<double> res(n);
        fprintf(stderr, "[dbg] reached before realize\n"); fflush(stderr);
        out.realize(res);   // compile + run once
        fprintf(stderr, "[dbg] realize done\n"); fflush(stderr);

        // validate vs host reference (identical algorithm)
        std::vector<double> xp(n, 0), xc(n, 0), xn(n), rr(n), zz(n);
        for (int kk = 0; kk < M; kk++) {
            applyA(xc, tmp); for (int i2 = 0; i2 < n; i2++) rr[i2] = bh[i2] - tmp[i2];
            applyMinv(rr, zz);
            for (int i2 = 0; i2 < n; i2++) xn[i2] = (1 + omh[kk]) * xc[i2] - omh[kk] * xp[i2] + alh[kk] * zz[i2];
            xp = xc; xc = xn;
        }
        double err = 0, en = 0, xerr = 0, xen = 0;
        for (int i2 = 0; i2 < n; i2++) {
            double e = res(i2) - xc[i2]; err += e * e; en += xc[i2] * xc[i2];
            double e2 = res(i2) - xe[i2]; xerr += e2 * e2; xen += xe[i2] * xe[i2];
        }
        printf("  Halide-inductive vs C++ ref: rel %.3e   vs exact: rel %.3e -> %s\n",
               std::sqrt(err / en), std::sqrt(xerr / xen),
               std::sqrt(err / en) < 1e-9 ? "PASS" : "FAIL");
    } catch (const Halide::CompileError &e) {
        printf("HALIDE CompileError: %s\n", e.what());
        return 2;
    } catch (const Halide::Error &e) {
        printf("HALIDE Error: %s\n", e.what());
        return 2;
    } catch (...) {
        printf("HALIDE ERROR (unknown catch-all)\n");
        return 2;
    }
    return 0;
}
