// Monte-Carlo pricing of an arithmetic-average Asian call, in Halide.
//
// Payoff depends on the AVERAGE price along each path -> the running average is
// an in-order consumer of the trajectory (no closed form; genuine path MC, the
// standard QuantLib DiscreteAveragingAsianOption benchmark). Under GBM each
// path is an exact log-step recurrence:
//   S_t = S_{t-1} * exp((r-0.5σ²)dt + σ√dt · Z_t),  A = mean_t S_t
//   price = e^{-rT} · mean_paths max(A - K, 0)
//
// S and its running sum are single-component scans over t (clean, unlike the
// coupled Kalman/ODE cases). B paths run in parallel; the path is an
// intermediate consumed by the running sum, so it folds to a 2-slice window and
// is never materialized.
//
// Build: g++ apps/montecarlo/asian_mc.cpp -O3 -march=native -Iinclude \
//        -Lbuild/src -lHalide -lpthread -ldl -o /tmp/amc -std=c++17
//        LD_LIBRARY_PATH=build/src /tmp/amc [B T]

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    int B = argc > 1 ? atoi(argv[1]) : 1000000;   // paths
    int T = argc > 2 ? atoi(argv[2]) : 128;        // averaging dates
    const double S0 = 100, K = 100, r = 0.05, sigma = 0.2, Tyears = 1.0;
    const double dt = Tyears / T;
    const float drift = (float)((r - 0.5 * sigma * sigma) * dt);
    const float vol = (float)(sigma * std::sqrt(dt));
    const float disc = (float)std::exp(-r * Tyears);

    // Shared standard-normal draws (so Halide and the C++ reference match).
    Buffer<float> Z(B, T);
    std::mt19937 rng(12345);
    std::normal_distribution<float> N(0.0f, 1.0f);
    for (int b = 0; b < B; b++)
        for (int t = 0; t < T; t++) Z(b, t) = N(rng);

    try {
        Var b("b"), t("t");

        // Path (inductive scan) and its running sum (inductive scan reading S).
        Func S(Float(32), "S"), cs(Float(32), "cs");
        S(b, t) = select(t <= 0, (float)S0 * exp(drift + vol * Z(b, 0)),
                         likely(S(b, t - 1) * exp(drift + vol * Z(b, t))));
        cs(b, t) = select(t <= 0, S(b, 0), likely(cs(b, t - 1) + S(b, t)));

        Func A("A");
        A(b) = cs(b, T - 1) * (1.0f / T);           // per-path average

        // Fold: keep only a 2-slice window of S and cs per path; the full path
        // is never stored.
        A.parallel(b).bound(b, 0, B);
        S.compute_at(A, b).store_at(A, b).fold_storage(t, 2);
        cs.compute_at(A, b).store_at(A, b).fold_storage(t, 2);

        A.compile_jit();
        Buffer<float> Ab(B);
        A.realize(Ab);

        auto price_from_A = [&](const float *a) {
            double s = 0;
            for (int i = 0; i < B; i++) s += std::max((double)a[i] - K, 0.0);
            return disc * s / B;
        };
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
        double t_hal = bench([&]() { A.realize(Ab); });
        double price_hal = price_from_A(Ab.data());

        // C++ reference (this TU is built -O3), single-thread.
        std::vector<float> Ac(B);
        double t_cpp = bench([&]() {
            for (int i = 0; i < B; i++) {
                float s = (float)S0 * std::exp(drift + vol * Z(i, 0));
                float sum = s;
                for (int t = 1; t < T; t++) { s *= std::exp(drift + vol * Z(i, t)); sum += s; }
                Ac[i] = sum / T;
            }
        });
        double price_cpp = price_from_A(Ac.data());

        printf("Arithmetic Asian call MC  B=%d paths  T=%d dates\n", B, T);
        printf("  params: S0=%.0f K=%.0f r=%.2f sigma=%.2f T=%.1f\n", S0, K, r, sigma, Tyears);
        printf("  Halide price = %.5f   (%.3f ms, %d threads)\n", price_hal, t_hal, 0);
        printf("  C++ price    = %.5f   (%.3f ms, 1 thread)\n", price_cpp, t_cpp);
        printf("  |Halide - C++| = %.3g\n", std::abs(price_hal - price_cpp));

        // Emit params for the QuantLib script.
        FILE *f = fopen("apps/montecarlo/params.txt", "w");
        if (f) { fprintf(f, "%g %g %g %g %g %d %d %.6f\n", S0, K, r, sigma, Tyears, T, B, price_hal); fclose(f); }
        return std::abs(price_hal - price_cpp) < 1e-2 ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
