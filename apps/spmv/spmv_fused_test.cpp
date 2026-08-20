// ELL SpMV fused with an in-order consumer: inductive vs non-inductive.
//
// Pipeline: y = A*x, then a running post-op z(i) = z(i-1) + y(i) (a prefix sum
// of the result -- stands in for the next in-order stage of a solver).
//
//   inductive : y is an inductive within-row sum; z is an inductive scan. The
//               whole chain fuses into one i-loop, y is never materialized
//               (folded to a scalar), z folds to a 2-element window.
//   noninduct : y is an RDom reduction that -- because an RDom can't be fused
//               into a recursive consumer -- must be fully materialized (R
//               floats) before z's RDom scan runs.
//
// Build: g++ apps/spmv/spmv_fused_test.cpp -Iinclude -Lbuild/src -lHalide
//        -lpthread -ldl -o /tmp/spmvf -std=c++17 ; LD_LIBRARY_PATH=build/src /tmp/spmvf

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

        // ---- inductive: SpMV sum + prefix-sum consumer, fully fused ----
        // y is inlined: z reads the row sum acc(K-1,i) directly. Two chained
        // inductive scans (acc over j, z over i), each folded to a tiny window.
        Func acc(Float(32), "acc"), z(Float(32), "z"), z_out("z_out");
        acc(j, i) = select(j <= 0, p(0, i), likely(acc(j - 1, i) + p(j, i)));
        z(i) = select(i <= 0, acc(K - 1, 0), likely(z(i - 1) + acc(K - 1, i)));
        z_out(i) = z(i);  // wrapper: inductive funcs can't be outputs
        acc.compute_at(z_out, i).store_at(z_out, Var::outermost()).fold_storage(j, 1);
        z.compute_at(z_out, i).store_at(z_out, Var::outermost()).fold_storage(i, 2);
        z_out.bound(i, 0, R);

        // ---- non-inductive: RDom reduce (materialized) + RDom prefix scan ----
        Func yr("yr"), zr("zr");
        yr(i) = 0.0f;
        yr(i) += p(rj, i);
        yr.compute_root();
        zr(i) = undef<float>();
        zr(0) = yr(0);
        RDom ri(1, R - 1);
        zr(ri) = zr(ri - 1) + yr(ri);

        if (getenv("SHOW_NEST")) { z_out.print_loop_nest(); return 0; }
        z_out.compile_jit();
        zr.compile_jit();
        Buffer<float> bi(R), br(R);
        z_out.realize(bi);
        zr.realize(br);

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
        double t_ind = bench(z_out, bi), t_rd = bench(zr, br);

        double err = 0, run = 0;
        for (int ii = 0; ii < R; ii++) {
            float g = 0;
            for (int jj = 0; jj < K; jj++) g += eval(jj, ii) * x(ecol(jj, ii));
            run += g;
            err = std::max({err, (double)std::abs(bi(ii) - run), (double)std::abs(br(ii) - run)});
        }

        printf("ELL SpMV + prefix-sum consumer  R=%d M=%d K=%d (N=%d)\n", R, M, K, R * K);
        printf("  inductive (fused, y not stored): %7.3f ms\n", t_ind);
        printf("  non-inductive (y materialized):  %7.3f ms\n", t_rd);
        printf("  max abs err %.3g -> %s\n", err, err < 1e-1 ? "PASS" : "FAIL");
        return err < 1e-1 ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
