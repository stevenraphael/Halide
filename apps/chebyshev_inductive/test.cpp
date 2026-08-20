// Runs the inductive and non-inductive Chebyshev solvers, checks they agree with
// each other and with a plain-C++ reference, and reports timing.

// Self-contained: the Chebyshev pipeline is built and JIT-compiled inline (from
// the same recurrence as chebyshev_inductive_generator.cpp), so no separate
// generator/AOT compilation step is needed.

#include "Halide.h"
#include "../support/bench_harness.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace Halide;

namespace {

// Must match the generator's M GeneratorParam.
constexpr int M = 60;

void host_matvec(const std::vector<double> &A, int n, const std::vector<double> &x,
                 std::vector<double> &y) {
    for (int i = 0; i < n; i++) y[i] = 0.0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) y[i] += A[(size_t)j * n + i] * x[j];
}

// A = M^T M + n I guarantees lambda_min >= n.
void make_spd(int n, std::vector<double> &A, std::vector<double> &b, std::vector<double> &x_exact) {
    uint64_t s = 1;
    auto rnd = [&] {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (double)(s >> 11) / (double)(1ULL << 53) * 2.0 - 1.0;
    };
    std::vector<double> Mm((size_t)n * n);
    for (auto &v : Mm) v = rnd();
    A.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int kk = 0; kk < n; kk++) acc += Mm[(size_t)i * n + kk] * Mm[(size_t)j * n + kk];
            A[(size_t)j * n + i] = acc + (i == j ? (double)n : 0.0);
        }
    x_exact.assign(n, 0.0);
    for (int i = 0; i < n; i++) x_exact[i] = rnd();
    b.assign(n, 0.0);
    host_matvec(A, n, x_exact, b);
}

double lambda_max_est(const std::vector<double> &A, int n, int iters = 60) {
    std::vector<double> v(n, 1.0), w(n);
    double lam = 0.0;
    for (int it = 0; it < iters; it++) {
        host_matvec(A, n, v, w);
        double nrm = 0.0;
        for (int i = 0; i < n; i++) nrm += w[i] * w[i];
        nrm = std::sqrt(nrm);
        if (nrm == 0) break;
        for (int i = 0; i < n; i++) v[i] = w[i] / nrm;
        lam = nrm;
    }
    return lam;
}

void make_coeffs(double lmin, double lmax, int m, std::vector<double> &alpha,
                 std::vector<double> &omega) {
    double d = 0.5 * (lmax + lmin), c = 0.5 * (lmax - lmin);
    alpha.assign(m, 0.0);
    omega.assign(m, 0.0);
    double a_prev = 0.0;
    for (int k = 0; k < m; k++) {
        if (k == 0) {
            alpha[k] = 1.0 / d;
            omega[k] = 0.0;
        } else {
            double beta = (c * a_prev * 0.5) * (c * a_prev * 0.5);
            alpha[k] = 1.0 / (d - beta / a_prev);
            omega[k] = alpha[k] * beta / a_prev;
        }
        a_prev = alpha[k];
    }
}

std::vector<double> host_chebyshev(const std::vector<double> &A, int n, const std::vector<double> &b,
                                   const std::vector<double> &alpha, const std::vector<double> &omega,
                                   int m) {
    std::vector<double> xp(n, 0.0), xc(n, 0.0), xn(n), Ax(n);
    for (int k = 0; k < m; k++) {
        host_matvec(A, n, xc, Ax);
        double a = alpha[k], w = omega[k];
        for (int i = 0; i < n; i++) xn[i] = (1.0 + w) * xc[i] - w * xp[i] + a * (b[i] - Ax[i]);
        xp = xc;
        xc = xn;
    }
    return xc;
}

double rel_error(const std::vector<double> &x, const std::vector<double> &y) {
    double e = 0, nn = 0;
    for (size_t i = 0; i < x.size(); i++) {
        double dd = x[i] - y[i];
        e += dd * dd;
        nn += y[i] * y[i];
    }
    return std::sqrt(e / (nn > 0 ? nn : 1.0));
}

// JIT build of the Chebyshev semi-iteration (ported from the generator). Both
// variants keep the same 2-D RDom (output component r.x, mat-vec component r.y)
// nested in the iteration index; they differ only in how the iterate history is
// stored: inductive folds column storage to 3 via fold_storage, non-inductive
// builds the same mod-3 ring by hand and materializes.
Func build_cheb(bool inductive, Buffer<double> &A, Buffer<double> &b,
                Buffer<double> &alpha, Buffer<double> &omega, int Miter) {
    Var t("t"), k("k");
    Expr n = A.dim(0).extent();
    Func x(Float(64), inductive ? "x_i" : "x_n");
    if (inductive) {
        Func X(Float(64), "X");
        X(t, k) = cast<double>(0);
        RDom r(0, n, 0, n, "r");
        Expr km1 = max(0, k - 1), km2 = max(0, k - 2);
        Expr once = (Expr(1.0) + omega(km1)) * X(r.x, km1) - omega(km1) * X(r.x, km2) +
                    alpha(km1) * b(r.x);
        X(r.x, k) = select(k <= 0, cast<double>(0),
                           X(r.x, k) + cast<double>(r.y == 0) * once -
                               alpha(km1) * A(r.x, r.y) * X(r.y, km1));
        RDom rk(0, Miter + 1, "rk");
        x(t) = cast<double>(0);
        x(t) += select(rk == Miter, X(t, rk), cast<double>(0));
        x.update(0).reorder(t, rk);
        X.compute_at(x, rk).store_root().fold_storage(k, 3);
        X.update(0).allow_race_conditions().vectorize(r.x, 4);
    } else {
        Func X(Float(64), "Xni");
        X(t, k) = cast<double>(0);
        RDom r(0, n, 0, n, 1, Miter, "r");
        Expr cur = r.z % 3, c1 = (r.z + 2) % 3, c2 = (r.z + 1) % 3, km1 = r.z - 1;
        Expr once = (Expr(1.0) + omega(km1)) * X(r.x, c1) - omega(km1) * X(r.x, c2) +
                    alpha(km1) * b(r.x);
        Expr matvec = alpha(km1) * A(r.x, r.y) * X(r.y, c1);
        X(r.x, cur) = select(r.y == 0, once - matvec, X(r.x, cur) - matvec);
        x(t) = X(t, Miter % 3);
        X.compute_root();
        X.update(0).allow_race_conditions().vectorize(r.x, 4);
    }
    return x;  // output bounds come from the realize target buffer
}

}  // namespace

int main(int argc, char **argv) {
    const int n = 256;

    std::vector<double> Ah, bh, x_exact;
    make_spd(n, Ah, bh, x_exact);
    double lmax = lambda_max_est(Ah, n) * 1.02;
    double lmin = (double)n * 0.98;  // A = M^T M + n I => lambda_min >= n
    std::vector<double> alpha_h, omega_h;
    make_coeffs(lmin, lmax, M, alpha_h, omega_h);
    std::vector<double> ref = host_chebyshev(Ah, n, bh, alpha_h, omega_h, M);

    // A is column-major (dim0 = row, fastest), matching make_spd.
    Buffer<double> A(Ah.data(), n, n);
    Buffer<double> b(bh.data(), n);
    Buffer<double> alpha(alpha_h.data(), M);
    Buffer<double> omega(omega_h.data(), M);
    Buffer<double> x_inductive(n), x_noninductive(n);

    Func f_ind = build_cheb(true, A, b, alpha, omega, M);
    Func f_non = build_cheb(false, A, b, alpha, omega, M);
    f_ind.compile_jit();
    f_non.compile_jit();
    hb::Stats s_ind = hb::bench([&]() { f_ind.realize(x_inductive); });
    hb::Stats s_non = hb::bench([&]() { f_non.realize(x_noninductive); });

    std::vector<double> xi(n), xn(n);
    for (int i = 0; i < n; i++) {
        xi[i] = x_inductive(i);
        xn[i] = x_noninductive(i);
    }

    double diff = rel_error(xi, xn);
    double err_i = rel_error(xi, x_exact), err_n = rel_error(xn, x_exact);
    double ref_i = rel_error(xi, ref), ref_n = rel_error(xn, ref);
    printf("inductive vs non-inductive: %g\n", diff);
    printf("error vs exact:   inductive %g   non-inductive %g\n", err_i, err_n);
    printf("error vs C++ ref: inductive %g   non-inductive %g\n", ref_i, ref_n);

    bool ok = !(diff > 1e-9 || err_i > 1e-6 || err_n > 1e-6 || ref_i > 1e-9 || ref_n > 1e-9);
    // Inductive folds the iterate history to 3 columns (O(3n)); non-inductive
    // materializes all M iterates (O(Mn)). On this small dense CPU case the two
    // are an honest tie -- reported as such.
    const double bytes_ind = (double)3 * n * 8;
    const double bytes_non = (double)M * n * 8;
    char note[128];
    snprintf(note, sizeof(note), "Chebyshev semi-iteration  n=%d M=%d  |  state fold %.0fx",
             n, M, hb::mem_ratio(bytes_non, bytes_ind));
    hb::print_spec_header("chebyshev", "host", note);
    hb::print_row("non-inductive (M iters materialized)", s_non,
                  (double)M / (s_non.median * 1e-3), "iter/s", bytes_non, err_n, ok);
    hb::print_row("inductive (fold -> 3 cols)", s_ind,
                  (double)M / (s_ind.median * 1e-3), "iter/s", bytes_ind, err_i, ok, "tie");

    if (diff > 1e-9 || err_i > 1e-6 || err_n > 1e-6 || ref_i > 1e-9 || ref_n > 1e-9) {
        printf("Chebyshev solvers disagree or did not converge!\n");
        return 1;
    }

    printf("Success!\n");
    return 0;
}
