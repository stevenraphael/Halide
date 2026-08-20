// ELL SpMV (y = A*x), inductive vs non-inductive.
//
// A is R x M in ELL layout: every row padded to K nonzeros, ell_col[j,i] /
// ell_val[j,i]. (Ragged CSR isn't expressible as one inductive func -- Halide
// can't bound a recursion whose base case is a data-dependent row start; ELL
// puts the base case at j<=0.)
//
//   inductive : within-row running sum acc(j,i)=acc(j-1,i)+p(j,i), consumed in
//               order along j, folded to a single scalar (fold_storage(j,1)).
//   noninduct : y(i) = sum_j p(j,i) via an RDom reduction.
//
// Build: g++ apps/spmv/spmv_test.cpp -Iinclude -Lbuild/src -lHalide -lpthread
//        -ldl -o /tmp/spmv -std=c++17 ; LD_LIBRARY_PATH=build/src /tmp/spmv

#include "Halide.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    int R = argc > 1 ? atoi(argv[1]) : 200000;
    int M = argc > 2 ? atoi(argv[2]) : 200000;
    int K = argc > 3 ? atoi(argv[3]) : 32;

    srand(2024);
    Buffer<int> ecol(K, R);
    Buffer<float> eval(K, R), x(M);
    for (int i = 0; i < R; i++)
        for (int j = 0; j < K; j++) {
            ecol(j, i) = rand() % M;
            eval(j, i) = (float)(rand() % 100) / 50.0f - 1.0f;
        }
    for (int j = 0; j < M; j++) x(j) = (float)(rand() % 100) / 50.0f - 1.0f;

    try {
        Var i("i"), j("j");
        RDom rj(0, K);

        Func p("p");
        p(j, i) = eval(j, i) * x(clamp(ecol(j, i), 0, M - 1));

        Func acc(Float(32), "acc"), y_ind("y_ind");
        acc(j, i) = select(j <= 0, p(0, i), likely(acc(j - 1, i) + p(j, i)));
        y_ind(i) = acc(K - 1, i);
        acc.compute_at(y_ind, i).store_at(y_ind, Var::outermost()).fold_storage(j, 1);

        Func y_rd("y_rd");
        y_rd(i) = 0.0f;
        y_rd(i) += p(rj, i);

        y_ind.compile_jit();
        y_rd.compile_jit();
        Buffer<float> bi(R), br(R);
        y_ind.realize(bi);
        y_rd.realize(br);

        auto bench = [&](Func f, Buffer<float> &b) {
            double best = 1e18;
            for (int t = 0; t < 20; t++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                f.realize(b);
                auto t1 = std::chrono::high_resolution_clock::now();
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            return best;
        };
        double t_ind = bench(y_ind, bi), t_rd = bench(y_rd, br);

        double err = 0;
        for (int ii = 0; ii < R; ii++) {
            float g = 0;
            for (int jj = 0; jj < K; jj++) g += eval(jj, ii) * x(ecol(jj, ii));
            err = std::max({err, (double)std::abs(bi(ii) - g), (double)std::abs(br(ii) - g)});
        }

        printf("ELL SpMV R=%d M=%d K=%d (N=%d)\n", R, M, K, R * K);
        printf("  inductive (folded sum): %7.3f ms\n", t_ind);
        printf("  non-inductive (RDom):   %7.3f ms\n", t_rd);
        printf("  max abs err %.3g -> %s\n", err, err < 1e-2 ? "PASS" : "FAIL");
        return err < 1e-2 ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
