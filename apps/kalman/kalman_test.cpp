// Batched 1-D constant-velocity Kalman filter in Halide, inductive vs a C++
// reference (the standard KF equations used by FilterPy/OpenCV/Eigen).
//
// State per tracker: x = [pos, vel] (2-vec), covariance P (2x2). The recurrence
// over time t carries (x, P) -- multi-valued state. As chained single-valued
// inductive funcs (xf over t, Pf over t) folded to a 2-slice window, P is never
// materialized over the whole sequence; only the current (x,P) is live. B
// independent trackers run in parallel.
//
// Model: F=[[1,dt],[0,1]], H=[1,0], Q=diag(q0,q1), R scalar. The covariance
// update is measurement-independent, so Pf needs no z; xf uses z and the gain
// derived from the predicted covariance.
//
// Build: g++ apps/kalman/kalman_test.cpp -Iinclude -Lbuild/src -lHalide
//        -lpthread -ldl -o /tmp/kf -std=c++17 ; LD_LIBRARY_PATH=build/src /tmp/kf

#include "Halide.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    int B = argc > 1 ? atoi(argv[1]) : 100000;   // number of trackers (parallel)
    int T = argc > 2 ? atoi(argv[2]) : 256;      // timesteps (recurrence)

    const float dt = 1.0f, R = 1.0f, q0 = 0.01f, q1 = 0.01f, p0 = 10.0f;

    // Measurements: noisy position of a target moving at constant velocity.
    srand(11);
    Buffer<float> z(B, T);
    for (int b = 0; b < B; b++) {
        float v = 0.5f + (rand() % 100) / 100.0f;
        float x0 = (float)(rand() % 1000);
        for (int t = 0; t < T; t++) {
            float noise = (float)(rand() % 200) / 100.0f - 1.0f;
            z(b, t) = x0 + v * t + noise;
        }
    }

    try {
        Var b("b"), t("t"), e("e"), c("c");

        // Non-inductive: RDom scan over t. Pf (covariance) and xf (state) are
        // update-def scans; the reduction variable serializes t and reads of the
        // func see the previous iteration's writes (RAW), so the coupled
        // multi-component recurrence is expressed correctly.
        Func Pf(Float(32), "Pf"), xf(Float(32), "xf");
        RDom rt(1, T - 1);

        Pf(e, b, t) = undef<float>();
        Pf(e, b, 0) = select(e == 0 || e == 3, p0, 0.0f);
        {
            Expr p00 = Pf(0, b, rt - 1), p01 = Pf(1, b, rt - 1),
                 p10 = Pf(2, b, rt - 1), p11 = Pf(3, b, rt - 1);
            Expr a00 = p00 + dt * p10, a01 = p01 + dt * p11, a10 = p10, a11 = p11;
            Expr Pm00 = a00 + dt * a01 + q0, Pm01 = a01, Pm10 = a10 + dt * a11, Pm11 = a11 + q1;
            Expr S = Pm00 + R, K0 = Pm00 / S, K1 = Pm10 / S;
            Expr nP00 = (1.0f - K0) * Pm00, nP01 = (1.0f - K0) * Pm01,
                 nP10 = Pm10 - K1 * Pm00, nP11 = Pm11 - K1 * Pm01;
            Pf(e, b, rt) = select(e == 0, nP00, e == 1, nP01, e == 2, nP10, nP11);
        }

        xf(c, b, t) = undef<float>();
        xf(c, b, 0) = select(c == 0, z(b, 0), 0.0f);
        {
            Expr p00 = Pf(0, b, rt - 1), p10 = Pf(2, b, rt - 1), p11 = Pf(3, b, rt - 1);
            Expr a00 = p00 + dt * p10, a01 = Pf(1, b, rt - 1) + dt * p11, a10 = p10, a11 = p11;
            Expr Pm00 = a00 + dt * a01 + q0, Pm10 = a10 + dt * a11;
            Expr S = Pm00 + R, K0 = Pm00 / S, K1 = Pm10 / S;
            Expr xm0 = xf(0, b, rt - 1) + dt * xf(1, b, rt - 1), xm1 = xf(1, b, rt - 1);
            Expr innov = z(b, rt) - xm0;
            xf(c, b, rt) = select(c == 0, xm0 + K0 * innov, xm1 + K1 * innov);
        }

        Func out("out");
        out(b, t) = xf(0, b, t);

        out.parallel(b).bound(b, 0, B).bound(t, 0, T);
        // Scan axis rt must be outer to the component axis (e/c inner).
        Pf.compute_at(out, b);
        Pf.update(1).reorder(e, rt);
        xf.compute_at(out, b);
        xf.update(1).reorder(c, rt);

        out.compile_jit();
        Buffer<float> hb(B, T);
        out.realize(hb);

        auto bench = [&](auto fn) {
            double best = 1e18;
            for (int i = 0; i < 10; i++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                fn();
                auto t1 = std::chrono::high_resolution_clock::now();
                best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            return best;
        };
        double t_hal = bench([&]() { out.realize(hb); });

        // C++ reference (standard KF equations) + timing + ground truth.
        std::vector<float> cb(B * T);
        double t_cpp = bench([&]() {
            for (int bb = 0; bb < B; bb++) {
                float x0 = z(bb, 0), x1 = 0.0f;
                float P00 = p0, P01 = 0, P10 = 0, P11 = p0;
                cb[bb * T + 0] = x0;
                for (int tt = 1; tt < T; tt++) {
                    float a00 = P00 + dt * P10, a01 = P01 + dt * P11, a10 = P10, a11 = P11;
                    float M00 = a00 + dt * a01 + q0, M01 = a01, M10 = a10 + dt * a11, M11 = a11 + q1;
                    float s = M00 + R, k0 = M00 / s, k1 = M10 / s;
                    float xm0 = x0 + dt * x1, xm1 = x1, in = z(bb, tt) - xm0;
                    x0 = xm0 + k0 * in; x1 = xm1 + k1 * in;
                    P00 = (1 - k0) * M00; P01 = (1 - k0) * M01;
                    P10 = M10 - k1 * M00; P11 = M11 - k1 * M01;
                    cb[bb * T + tt] = x0;
                }
            }
        });

        if (getenv("DUMP")) {
            printf("t: z  halide  cpp\n");
            for (int tt = 0; tt < std::min(T, 8); tt++)
                printf("%d: %.4f  %.4f  %.4f\n", tt, z(0, tt), hb(0, tt), cb[0 * T + tt]);
        }
        double err = 0;
        for (int bb = 0; bb < B; bb++)
            for (int tt = 0; tt < T; tt++)
                err = std::max(err, (double)std::abs(hb(bb, tt) - cb[bb * T + tt]));

        printf("Kalman (1D CV)  B=%d trackers  T=%d steps\n", B, T);
        printf("  Halide non-inductive (RDom scan):   %7.3f ms\n", t_hal);
        printf("  C++ reference (FilterPy-style eqs): %7.3f ms\n", t_cpp);
        printf("  max abs err %.3g -> %s\n", err, err < 1e-2 ? "PASS" : "FAIL");
        return err < 1e-2 ? 0 : 1;
    } catch (const Halide::Error &ex) {
        printf("HALIDE ERROR: %s\n", ex.what());
        return 2;
    }
}
