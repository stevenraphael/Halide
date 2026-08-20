// Minimal-overhead, general-purpose-but-genuinely-optimized third-party
// comparison: Boost.Odeint's implicit_euler stepper (header-only C++
// template library -- fully inlined at -O3, no virtual dispatch, no
// Python/interpreter overhead) called in a raw fixed-step C++ loop (no
// adaptive step controller -- do_step() is called with a constant dt), on
// the SAME damped-pendulum problem as pendulum_mujoco.cpp (derived from
// matching MuJoCo's own qfrc_bias/mass values, see pendulum_mujoco_check.py).
//
// Note: Boost.Odeint's implicit_euler does a FULL Newton solve on the whole
// nonlinear system (gravity term included), whereas MuJoCo's own "implicit"
// integrator only linearizes/implicitly-treats the damping term (gravity
// stays explicit). So this is not expected to bit-match
// pendulum_mujoco.cpp's trajectory -- it's an independent, honestly-different
// discretization of the same physical system, used purely to measure a
// genuinely optimized (no interpreter, no unrelated framework overhead)
// general-purpose C++ library's raw per-step cost against Halide's.
//
// Build: g++ apps/ode_implicit/ode_implicit_odeint.cpp -O3 -march=native
//        -fopenmp -o /tmp/odeint_pend -std=c++17
//        ; /tmp/odeint_pend [B T]

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using state_type = boost::numeric::ublas::vector<double>;
using matrix_type = boost::numeric::ublas::matrix<double>;

static const double M_g = 0.08573594936708857;
static const double damping_g = 2.0;
static const double A_g = 2.4525;

struct pend_rhs {
    void operator()(const state_type &y, state_type &dydt, double /*t*/) const {
        dydt(0) = y(1);
        dydt(1) = (A_g * std::cos(y(0)) - damping_g * y(1)) / M_g;
    }
};

struct pend_jac {
    void operator()(const state_type &y, matrix_type &J, double /*t*/) const {
        J(0, 0) = 0.0;
        J(0, 1) = 1.0;
        J(1, 0) = -A_g * std::sin(y(0)) / M_g;
        J(1, 1) = -damping_g / M_g;
    }
};

int main(int argc, char **argv) {
    int B = argc > 1 ? atoi(argv[1]) : 256;
    int T = argc > 2 ? atoi(argv[2]) : 20000;
    const double h = 0.001;

    std::vector<double> theta0(B);
    srand(3);
    for (int b = 0; b < B; b++) theta0[b] = (rand() % 300) / 100.0 - 1.5;

    std::vector<double> obs((size_t)B * T);

    using namespace boost::numeric::odeint;
    auto t0 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < B; b++) {
        state_type y(2);
        y(0) = theta0[b];
        y(1) = 0.0;
        implicit_euler<double> stepper;
        obs[(size_t)b * T] = y(0);
        double t = 0.0;
        for (int step = 1; step < T; step++) {
            stepper.do_step(std::make_pair(pend_rhs{}, pend_jac{}), y, t, h);
            t += h;
            obs[(size_t)b * T + step] = y(0);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("Boost.Odeint implicit_euler (fixed step, raw C++ loop, damped pendulum)  B=%d T=%d h=%.4f\n", B, T, h);
    printf("  time: %8.3f ms\n", ms);

    FILE *fp = fopen("apps/ode_implicit/pend_params.txt", "r");
    if (fp) {
        int hb, ht; double hal_ms;
        if (fscanf(fp, "%d %d %lf", &hb, &ht, &hal_ms) == 3) {
            printf("  (Halide inductive, from last pendulum_mujoco run: %8.3f ms, %.1fx vs Boost.Odeint)\n",
                   hal_ms, ms / hal_ms);
        }
        fclose(fp);
    }

    auto dump = [](const char *p, const double *d, size_t n) {
        FILE *f = fopen(p, "wb"); if (f) { fwrite(d, sizeof(double), n, f); fclose(f); } };
    dump("apps/ode_implicit/odeint_pend_obs.bin", obs.data(), (size_t)B * T);
    return 0;
}
