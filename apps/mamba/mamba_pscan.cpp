// Mamba selective-SSM core via a TWO-STAGE blocked scan (sqrt(T) x sqrt(T)),
// the Mamba-2 / SSD chunking structure -- as opposed to the single sequential
// scan in mamba.cpp. Diagonal recurrence  h_t = a_t h_{t-1} + b_t x_t  (state n,
// channel d). Split t into C chunks of length L (~sqrt(T) each):
//   Stage 1 (intra-chunk, INDEPENDENT across chunks -> parallel):
//       PA(k,j) = prod_{i<=j} a         (multiplier on the chunk carry-in)
//       PB(k,j) = local h with carry-in 0 = a_j PB(k,j-1) + B_j
//   Stage 2 (inter-chunk carry, the SHORT sequential critical path, length C):
//       carry(k) = cA(k-1) carry(k-1) + cB(k-1),  cA=PA(k,L-1), cB=PB(k,L-1)
//   Stage 3 (apply carry, parallel):   h(k,j) = PA(k,j) carry(k) + PB(k,j)
//   y_t = sum_n c_t h_t.
// Each of the three scans is written EITHER as an inductive func (Var scan axis +
// select + likely) OR non-inductively (RDom scan). NOTE this exposes the rule:
// PA/PB are consumed IN FULL by stage 3, so they cannot fold -> inductive has
// nothing to save here (expect ~tie), unlike mamba.cpp's foldable trajectory.
//
// Build: g++ apps/mamba/mamba_pscan.cpp -O3 -march=native -fopenmp -Iinclude
//        -Lbuild/src -lHalide -lpthread -ldl -o /tmp/mps -std=c++17
//        LD_LIBRARY_PATH=build/src /tmp/mps [N D T]

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 16;
    int D = argc > 2 ? atoi(argv[2]) : 4096;
    int T = argc > 3 ? atoi(argv[3]) : 512;
    int L = (int)std::ceil(std::sqrt((double)T));  // chunk length ~ sqrt(T)
    int C = (T + L - 1) / L;                       // number of chunks

    Buffer<float> a(D, N, T), b(N, T), c(N, T), x(D, T);
    srand(5);
    for (int t = 0; t < T; t++) {
        for (int n = 0; n < N; n++) {
            b(n, t) = (rand() % 200) / 100.0f - 1.0f;
            c(n, t) = (rand() % 200) / 100.0f - 1.0f;
            for (int d = 0; d < D; d++)
                a(d, n, t) = 0.9f + (rand() % 99) / 1000.0f;
        }
        for (int d = 0; d < D; d++)
            x(d, t) = (rand() % 200) / 100.0f - 1.0f;
    }

    auto build = [&](bool inductive) -> Func {
        Var d("d"), n("n"), k("k"), j("j"), t("t");
        // Padded chunk view: t = k*L + j; pad with a=1, B=0 past T (identity map).
        Func gt("gt"), a_pad("a_pad"), B_pad("B_pad");
        Expr gtt = k * L + j;
        Expr valid = gtt < T;
        Expr ci = clamp(gtt, 0, T - 1);
        a_pad(d, n, k, j) = select(valid, a(d, n, ci), 1.0f);
        B_pad(d, n, k, j) = select(valid, b(n, ci) * x(d, ci), 0.0f);

        // Inductive funcs need an EXPLICIT type: the likely()-wrapped self-call
        // has unknown type, and select's type inference can't recurse through it.
        Func PA(Float(32), "PA"), PB(Float(32), "PB"), carry(Float(32), "carry");
        Func h("h");
        if (inductive) {
            // Stage 1: two inductive scans over the within-chunk index j.
            PA(d, n, k, j) = select(j <= 0, a_pad(d, n, k, 0),
                                    likely(a_pad(d, n, k, j) * PA(d, n, k, j - 1)));
            PB(d, n, k, j) = select(j <= 0, B_pad(d, n, k, 0),
                                    likely(a_pad(d, n, k, j) * PB(d, n, k, j - 1) + B_pad(d, n, k, j)));
            // Stage 2: inductive carry scan over the chunk index k.
            carry(d, n, k) = select(k <= 0, 0.0f,
                                    likely(PA(d, n, k - 1, L - 1) * carry(d, n, k - 1) + PB(d, n, k - 1, L - 1)));
        } else {
            // Stage 1: non-inductive RDom scans over j (RVar rj = scan axis).
            PA(d, n, k, j) = a_pad(d, n, k, j);
            PB(d, n, k, j) = B_pad(d, n, k, j);
            RDom rj(1, L - 1);
            PA(d, n, k, rj) = a_pad(d, n, k, rj) * PA(d, n, k, rj - 1);
            PB(d, n, k, rj) = a_pad(d, n, k, rj) * PB(d, n, k, rj - 1) + B_pad(d, n, k, rj);
            // Stage 2: non-inductive carry scan over k (RVar rk).
            carry(d, n, k) = 0.0f;
            RDom rk(1, C - 1);
            carry(d, n, rk) = PA(d, n, rk - 1, L - 1) * carry(d, n, rk - 1) + PB(d, n, rk - 1, L - 1);
        }
        // Stage 3 + per-step consumer.
        h(d, n, k, j) = PA(d, n, k, j) * carry(d, n, k) + PB(d, n, k, j);
        Func y("y");
        Expr kk = t / L, jj = t % L;
        Expr acc = 0.0f;
        for (int nn = 0; nn < N; nn++)
            acc += c(nn, t) * h(d, nn, kk, jj);
        y(d, t) = acc;

        // Schedule: channel d is the wide vectorize axis; chunks k are independent
        // (parallel). PA/PB/carry compute_root (stage 3 reads them in full, so no
        // fold is possible either way -- this is the point of the comparison).
        const int V = 8;
        Var do_("do"), di("di");
        y.bound(d, 0, D).bound(t, 0, T).split(d, do_, di, V).reorder(di, t, do_).vectorize(di).parallel(do_);
        for (Func f : {PA, PB}) {
            f.reorder_storage(d, n, k, j).compute_root().vectorize(d, V).parallel(k);
            if (!inductive) f.update(0).reorder(d, n, k).vectorize(d, V).parallel(k);
        }
        carry.reorder_storage(d, n, k).compute_root().vectorize(d, V);
        if (!inductive) carry.update(0).vectorize(d, V);
        h.compute_at(y, t).vectorize(d, V);
        return y;
    };

    try {
        Func yi = build(true), yn = build(false);
        yi.compile_jit();
        yn.compile_jit();
        Buffer<float> ri(D, T), rn(D, T);
        yi.realize(ri);
        yn.realize(rn);

        auto bench = [&](Func f, Buffer<float> &bb) {
            double best = 1e18;
            for (int q = 0; q < 5; q++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                f.realize(bb);
                auto t1 = std::chrono::high_resolution_clock::now();
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            return best;
        };
        double ti = bench(yi, ri), tn = bench(yn, rn);

        // C++ sequential streaming reference.
        std::vector<float> cy((size_t)D * T);
        const float *ap = a.data(), *bp = b.data(), *cp = c.data(), *xp = x.data();
#pragma omp parallel
        {
            std::vector<float> hs(N);
#pragma omp for schedule(static)
            for (int dd = 0; dd < D; dd++) {
                for (int nn = 0; nn < N; nn++)
                    hs[nn] = bp[nn] * xp[dd];
                float y0 = 0;
                for (int nn = 0; nn < N; nn++)
                    y0 += cp[nn] * hs[nn];
                cy[(size_t)dd * T] = y0;
                for (int t = 1; t < T; t++) {
                    float xdt = xp[(size_t)dd + (size_t)t * D], yy = 0;
                    for (int nn = 0; nn < N; nn++) {
                        float av = ap[(size_t)dd + (size_t)D * (nn + (size_t)N * t)];
                        hs[nn] = av * hs[nn] + bp[nn + (size_t)N * t] * xdt;
                        yy += cp[nn + (size_t)N * t] * hs[nn];
                    }
                    cy[(size_t)dd * T + t] = yy;
                }
            }
        }

        double err = 0;
        bool bad = false;
        for (int dd = 0; dd < D; dd++)
            for (int t = 0; t < T; t++) {
                float ai = ri(dd, t), cn = rn(dd, t), g = cy[(size_t)dd * T + t];
                if (std::isnan(ai) || std::isnan(cn)) bad = true;
                err = std::max({err, (double)std::abs(ai - g), (double)std::abs(cn - g)});
            }
        printf("Mamba TWO-STAGE (sqrt) scan  N=%d D=%d T=%d  (L=%d C=%d)\n", N, D, T, L, C);
        printf("  inductive scans:      %8.3f ms\n", ti);
        printf("  non-inductive scans:  %8.3f ms\n", tn);
        printf("  max abs err %.3g%s -> %s\n", err, bad ? " (NaN)" : "",
               (!bad && err < 1e-2) ? "PASS" : "FAIL");
        return (!bad && err < 1e-2) ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
