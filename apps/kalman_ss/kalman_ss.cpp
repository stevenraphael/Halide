// Batched steady-state Kalman filter, arbitrary N-dim state, per-step consumer.
//   x_t = A x_{t-1} + K z_t     (A = (I-KH)F precomputed steady-state form)
//   y_t = C . x_t               (per-step output, consumed every step)
// A is dense N x N -> x_t(i) reads the WHOLE previous state x_{t-1} via matvec.
// Inductive: single func, inline matvec of the PAST slice (scan axis is a Var so
// the sum() over j is a single reduction domain). Non-inductive: 2D-RDom scan.
// B independent filters (batch). Reference: C++ -O3 streaming.
//
// Build: g++ apps/kalman_ss/kalman_ss.cpp -O3 -march=native -Iinclude -Lbuild/src
//        -lHalide -lpthread -ldl -o /tmp/kss -std=c++17 ; LD_LIBRARY_PATH=build/src /tmp/kss [N B T]

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int N = argc > 1 ? atoi(argv[1]) : 32;      // state dim (arbitrary)
    int B = argc > 2 ? atoi(argv[2]) : 4096;     // batch of filters
    int T = argc > 3 ? atoi(argv[3]) : 512;      // timesteps

    // Stable dense A (rows scaled so spectral radius < 1), K, C, measurements z.
    Buffer<float> A(N, N), K(N), C(N), z(B, T);
    srand(4);
    for (int i = 0; i < N; i++) {
        float s = 0;
        for (int j = 0; j < N; j++) { A(j, i) = ((rand() % 200) / 100.0f - 1.0f); s += std::abs(A(j, i)); }
        for (int j = 0; j < N; j++) A(j, i) *= 0.9f / (s + 1e-6f);   // row-sum -> 0.9 (stable)
        K(i) = (rand() % 100) / 100.0f;
        C(i) = (rand() % 100) / 100.0f;
    }
    for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) z(b, t) = (rand() % 200) / 100.0f - 1.0f;

    auto build = [&](bool inductive) -> Func {
        Var b("b"), t("t");
        // Manually-unrolled matvec, organized as ONE Tuple-valued Func (N
        // float components) rather than N separate Funcs: N separate mutually
        // -referencing Funcs hit a hard Halide restriction (a pure/update
        // definition can't call a *different* Func before that Func has its
        // own definition, and with a dense matrix every x_i depends on every
        // x_j, so no definition order works). A single Func's Tuple self-
        // reference doesn't have that problem -- x(b, t-1) is the same Func,
        // so it's always allowed, and extracting a different component per
        // output index is exactly the same "constant-indexed dot product"
        // computation as the separate-Funcs version would have done.
        //
        // This C++-side unroll of the sum over j is not an optional
        // optimization -- it's necessitated by Halide's inductive-function
        // model, which requires every self-reference to sit lexically inside
        // a select() (so the base case and monotonic-decrease proof can be
        // checked). An RDom-driven "+=" reduction can't satisfy that: it
        // either needs its own accumulation target's position to be part of
        // the same RDom (the combined-RDom-over-(i,j) form, which then needs
        // allow_race_conditions() + vectorize-over-RVar and is measurably
        // slower), or it has no select() at all (rejected outright: "contains
        // an inductive function reference outside of a select operation
        // value"). Building the whole per-output select() expression via a
        // host-side loop is the only way to keep i a genuine pure Var while
        // still satisfying that structural requirement.
        std::vector<Type> types(N, Float(32));
        Func x(types, inductive ? "x_ind" : "x_mat");

        auto term = [&](int out_i, Expr tt, const std::vector<Expr> &prev) {
            Expr acc = K(out_i) * z(b, tt);
            for (int j = 0; j < N; j++) acc += A(j, out_i) * prev[j];
            return acc;
        };

        // NOTE: extract tuple components with FuncRef::operator[] (element by
        // element), not by constructing a Tuple(FuncRef) -- the latter has an
        // eager has_pure_definition() check with no self-reference exemption
        // (unlike the more permissive, self-reference-aware check that runs
        // later when the definition itself is assigned), so it rejects the
        // self-call before x has been given a definition at all.
        if (inductive) {
            std::vector<Expr> prev_vals(N);
            for (int j = 0; j < N; j++) prev_vals[j] = x(b, t - 1)[j];
            std::vector<Expr> vals(N);
            for (int out_i = 0; out_i < N; out_i++) {
                Expr base = K(out_i) * z(b, t);
                vals[out_i] = select(t <= 0, base, likely(term(out_i, t, prev_vals)));
            }
            x(b, t) = Tuple(vals);
        } else {
            // FAIR non-inductive: same manually-unrolled matvec as the
            // inductive version, as an RDom-over-time scan (so it can't
            // fold -- only difference from inductive is materialize vs
            // fold). Pure def's LHS args must all be plain Vars, so (as in
            // kalman_ll.cpp) it initializes every t to the base value; the
            // update then overwrites t in [1, T-1] with the full matvec.
            std::vector<Expr> base_vals(N);
            for (int out_i = 0; out_i < N; out_i++) base_vals[out_i] = K(out_i) * z(b, t);
            x(b, t) = Tuple(base_vals);
            RDom rt(1, T - 1, "rt");
            std::vector<Expr> prev_vals(N);
            for (int j = 0; j < N; j++) prev_vals[j] = x(b, rt - 1)[j];
            std::vector<Expr> vals(N);
            for (int out_i = 0; out_i < N; out_i++) vals[out_i] = term(out_i, rt, prev_vals);
            x(b, rt) = Tuple(vals);
        }

        // Per-step consumer: y(b,t) = sum_i C(i) x(b,t)[i]. Manual unroll
        // (pure func, constant tuple index) so an inductive x can fuse into
        // it (like rnn_cascade).
        Func y("y");
        Expr acc = 0.0f;
        for (int ii = 0; ii < N; ii++) acc += C(ii) * x(b, t)[ii];
        y(b, t) = acc;
        // Batch is the parallel + vectorizable axis (independent filters); time
        // is the serial scan axis. Both versions: vectorize over batch, parallel
        // over batch tiles.
        const int V = 8;
        Var bo("bo"), bi("bi");
        y.bound(b, 0, B).bound(t, 0, T)
         .split(b, bo, bi, V).reorder(bi, t, bo).vectorize(bi).parallel(bo);
        // Store x with batch INNERMOST so vectorizing over b is a contiguous SIMD
        // load.
        if (inductive) {
            x.reorder_storage(b, t)
             .compute_at(y, t).store_at(y, bo).fold_storage(t, 2).vectorize(b, V);
        } else if (getenv("MATROOT")) {
            x.reorder_storage(b, t).compute_root().vectorize(b, V);
            x.update(0).vectorize(b, V);
        } else {
            x.reorder_storage(b, t).compute_at(y, bo).vectorize(b, V);
            x.update(0).vectorize(b, V);
        }
        return y;
    };

    try {
        Func yi = build(true), yn = build(false);
        yi.compile_jit(); yn.compile_jit();
        Buffer<float> ri(B, T), rn(B, T);
        yi.realize(ri); yn.realize(rn);

        auto bench = [&](Func f, Buffer<float> &bb) {
            double best = 1e18;
            for (int k = 0; k < 5; k++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                f.realize(bb);
                auto t1 = std::chrono::high_resolution_clock::now();
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            return best;
        };
        double ti = bench(yi, ri), tn = bench(yn, rn);

        // C++ -O3 streaming reference (window-1 state, emits y each step).
        std::vector<float> cy((size_t)B * T);
        double tc = 0;
        {
            // Raw pointers -- Halide Buffer::operator() in a hot OpenMP loop is
            // slow and thread-hostile. Layouts: A(j,i)=Ap[j+i*N], z(b,t)=zp[b+t*B].
            const float *Ap = A.data(), *Kp = K.data(), *Cp = C.data(), *zp = z.data();
            auto t0 = std::chrono::high_resolution_clock::now();
            #pragma omp parallel
            {
            std::vector<float> xp(N), xn(N);   // allocated ONCE per thread
            #pragma omp for schedule(static)
            for (int b = 0; b < B; b++) {
                for (int ii = 0; ii < N; ii++) xp[ii] = Kp[ii] * zp[(size_t)b];
                float y0 = 0; for (int ii = 0; ii < N; ii++) y0 += Cp[ii] * xp[ii];
                cy[(size_t)b * T] = y0;
                for (int t = 1; t < T; t++) {
                    float zbt = zp[(size_t)b + (size_t)t * B];
                    for (int ii = 0; ii < N; ii++) {
                        float s = 0; const float *Arow = Ap + (size_t)ii * N;
                        for (int j = 0; j < N; j++) s += Arow[j] * xp[j];
                        xn[ii] = s + Kp[ii] * zbt;
                    }
                    float yy = 0; for (int ii = 0; ii < N; ii++) yy += Cp[ii] * xn[ii];
                    cy[(size_t)b * T + t] = yy;
                    xp.swap(xn);
                }
            }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            tc = std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        double err = 0; bool bad = false;
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) {
            float a = ri(b, t), c = rn(b, t), g = cy[(size_t)b * T + t];
            if (std::isnan(a) || std::isnan(c)) bad = true;
            err = std::max({err, (double)std::abs(a - g), (double)std::abs(c - g)});
        }
        double traj = (double)N * B * T * 4 / (1024.0 * 1024.0);
        printf("Steady-state Kalman  N=%d B=%d T=%d  (trajectory %.0f MB)\n", N, B, T, traj);
        printf("  inductive Halide (fold state->2):  %8.3f ms\n", ti);
        printf("  non-inductive Halide (per-batch):  %8.3f ms\n", tn);
        printf("  (C++ sanity ref, ignore timing):   %8.3f ms\n", tc);
        printf("  max abs err %.3g%s -> %s\n", err, bad ? " (NaN)" : "", (!bad && err < 1e-2) ? "PASS" : "FAIL");

        // Dump A, K, C, z, and inductive y for the NumPy/BLAS third-party bench.
        auto dump = [](const char *p, const float *d, size_t n) {
            FILE *f = fopen(p, "wb"); if (f) { fwrite(d, sizeof(float), n, f); fclose(f); } };
        dump("apps/kalman_ss/A.bin", A.data(), (size_t)N * N);
        dump("apps/kalman_ss/K.bin", K.data(), N);
        dump("apps/kalman_ss/C.bin", C.data(), N);
        dump("apps/kalman_ss/z.bin", z.data(), (size_t)B * T);
        std::vector<float> yflat((size_t)B * T);
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) yflat[(size_t)b * T + t] = ri(b, t);
        dump("apps/kalman_ss/y.bin", yflat.data(), (size_t)B * T);
        FILE *fp = fopen("apps/kalman_ss/params.txt", "w");
        if (fp) { fprintf(fp, "%d %d %d %.6f\n", N, B, T, ti); fclose(fp); }
        return (!bad && err < 1e-2) ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
