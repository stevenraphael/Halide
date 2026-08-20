// 1-D heat equation by method-of-lines: dy/dt = A y, A = (1/dx^2)*Laplacian
// (Dirichlet BCs). Integrated with 2-step Adams-Bashforth. This is the coupled
// vector-per-timestep recurrence: y_n[d] needs the WHOLE previous slice y_{n-1}
// via the matvec, so the time axis folds to a small window while the D-vector
// per slice is retained.
//
// Consumer = FINAL field y(:, T-1) ("evolve to time T, report the state"), so
// only the last slice is read and the trajectory folds. Time folded to 4.
// Built as a single 2D-RDom scan (r.x = matvec index, r.y = time) -- no tuples,
// no separate RHS func. B independent initial conditions run in parallel.
//
// Compared to a C++ -O3 reference (exact, same arithmetic) and validated
// against SciPy solve_ivp (see heat_scipy.py).
//
// Build: g++ apps/heat/heat_mol.cpp -O3 -march=native -Iinclude -Lbuild/src
//        -lHalide -lpthread -ldl -o /tmp/heat -std=c++17
//        LD_LIBRARY_PATH=build/src /tmp/heat [D B T]

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    int D = argc > 1 ? atoi(argv[1]) : 128;      // grid points
    int B = argc > 2 ? atoi(argv[2]) : 4096;     // initial conditions
    int T = argc > 3 ? atoi(argv[3]) : 512;      // timesteps
    const double dx = 1.0 / (D + 1);
    const double CFL = 0.2;                       // AB2 real-axis stability
    const float dt = (float)(CFL * dx * dx);
    const double tfinal = (T - 1) * (double)dt;

    // A = (1/dx^2) * tridiagonal Laplacian (dense, general coupled matvec).
    Buffer<float> A(D, D), y0(D, B);
    float inv = (float)(1.0 / (dx * dx));
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            A(j, i) = (i == j) ? -2.0f * inv
                               : ((i == j + 1 || j == i + 1) ? inv : 0.0f);
    srand(3);
    for (int bb = 0; bb < B; bb++)
        for (int i = 0; i < D; i++) y0(i, bb) = (float)(rand() % 200) / 100.0f - 1.0f;

    try {
        Var d("d"), b("b"), n("n");
        Func y(Float(32), "y");
        RDom r(0, D, 1, T - 1);
        Expr rk = r.x, rn = r.y;
        y(d, b, n) = undef<float>();
        y(d, b, 0) = y0(d, b);
        Expr carry = select(rk == 0, y(d, b, rn - 1), y(d, b, rn));
        Expr term = dt * A(d, rk) *
                    (1.5f * y(rk, b, rn - 1) - 0.5f * y(rk, b, clamp(rn - 2, 0, T - 1)));
        y(d, b, rn) = carry + term;

        // PER-STEP consumer: total heat at every timestep, H(b,t)=sum_d y(d,b,t).
        // It reads every slice, so the RDom scan can't fuse into it and y must be
        // materialized as the full O(D*B*T) trajectory.
        RDom rd(0, D);
        Func H("H");
        H(b, n) = 0.0f;
        H(b, n) += y(rd, b, n);

        // Non-inductive: y is the full materialized trajectory (no fold).
        y.compute_root();
        y.update(1).reorder(r.x, d, r.y, b).parallel(b);
        H.compute_root().parallel(n);
        H.bound(b, 0, B).bound(n, 0, T);

        double traj_mb = (double)D * B * T * sizeof(float) / (1024.0 * 1024.0);
        printf("  [non-inductive materializes trajectory: %.0f MB]\n", traj_mb);

        H.compile_jit();
        Buffer<float> hf(B, T);
        H.realize(hf);

        auto bench = [&](auto fn) {
            double best = 1e18;
            for (int i = 0; i < 5; i++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                fn();
                auto t1 = std::chrono::high_resolution_clock::now();
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            return best;
        };
        double t_hal = bench([&]() { H.realize(hf); });

        // C++ -O3 reference: STREAMS (ping-pong window 3), accumulating the same
        // per-step total heat -- never materializes the trajectory. This is what
        // the non-inductive Halide version is *forced* to materialize instead.
        std::vector<float> cH((size_t)B * T);
        double t_cpp = bench([&]() {
            std::vector<float> yn1(D), yn2(D), yc(D), f1(D), f2(D);
            for (int bb = 0; bb < B; bb++) {
                for (int i = 0; i < D; i++) yn1[i] = yn2[i] = y0(i, bb);
                float h0 = 0; for (int i = 0; i < D; i++) h0 += yn1[i];
                cH[(size_t)bb * T + 0] = h0;
                for (int nn = 1; nn < T; nn++) {
                    for (int i = 0; i < D; i++) {
                        float s1 = 0, s2 = 0;
                        for (int k = 0; k < D; k++) { s1 += A(k, i) * yn1[k]; s2 += A(k, i) * yn2[k]; }
                        f1[i] = s1; f2[i] = s2;
                    }
                    float hh = 0;
                    for (int i = 0; i < D; i++) { yc[i] = yn1[i] + dt * (1.5f * f1[i] - 0.5f * f2[i]); hh += yc[i]; }
                    yn2 = yn1; yn1 = yc;
                    cH[(size_t)bb * T + nn] = hh;
                }
            }
        });

        double err = 0; bool bad = false;
        for (int bb = 0; bb < B; bb++)
            for (int nn = 0; nn < T; nn++) {
                float hv = hf(bb, nn), cv = cH[(size_t)bb * T + nn];
                if (std::isnan(hv) || std::isinf(hv)) bad = true;
                err = std::max(err, (double)std::abs(hv - cv));
            }

        printf("Heat eq (method of lines, AB2), per-step consumer H(b,t)=sum_d y\n");
        printf("  D=%d B=%d T=%d  dt=%.3g tfinal=%.4f\n", D, B, T, dt, tfinal);
        printf("  Halide non-inductive (materializes traj, 22 thr): %8.3f ms\n", t_hal);
        printf("  C++ -O3 (streams, no trajectory, 1 thr):          %8.3f ms\n", t_cpp);
        printf("  |Halide - C++| = %.3g%s\n", err, bad ? " (NaN/Inf!)" : "");
        return (!bad && err < 1e-2) ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
