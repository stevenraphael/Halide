// Decisive test: can a SINGLE inductive func express a coupled scan where each
// step reads the whole previous slice (cross-component)? Euler on dy/dt = A y.
//   y(d,b,n) = y(d,b,n-1) + dt * sum_k A(d,k) y(k,b,n-1)
// Var scan axis n (so the inline matvec sum has one reduction domain); the
// self-reference reads the full previous slice y(:,b,n-1). Check vs C++.

#include "Halide.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    int D = argc > 1 ? atoi(argv[1]) : 8;
    int B = argc > 2 ? atoi(argv[2]) : 4;
    int T = argc > 3 ? atoi(argv[3]) : 8;
    const float dt = 0.1f;

    Buffer<float> A(D, D), y0(D, B);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++)
            A(j, i) = (i == j) ? -2.0f : ((i == j + 1 || j == i + 1) ? 1.0f : 0.0f);
    srand(1);
    for (int b = 0; b < B; b++)
        for (int i = 0; i < D; i++) y0(i, b) = (float)(rand() % 100) / 100.0f;

    try {
        Var d("d"), b("b"), n("n");
        RDom rk(0, D);
        Func y(Float(32), "y");
        Expr mv = sum(A(d, rk) * y(rk, b, n - 1));
        y(d, b, n) = select(n <= 0, y0(d, b), likely(y(d, b, n - 1) + dt * mv));

        Func yf("yf");
        yf(d, b) = y(d, b, T - 1);
        yf.bound(d, 0, D).bound(b, 0, B);
        y.compute_at(yf, b).store_at(yf, b).fold_storage(n, 2);

        Buffer<float> hf(D, B);
        yf.realize(hf);

        // C++ reference.
        double err = 0;
        std::vector<float> yv(D), yn(D);
        for (int bb = 0; bb < B; bb++) {
            for (int i = 0; i < D; i++) yv[i] = y0(i, bb);
            for (int t = 1; t < T; t++) {
                for (int i = 0; i < D; i++) {
                    float s = 0; for (int k = 0; k < D; k++) s += A(k, i) * yv[k];
                    yn[i] = yv[i] + dt * s;
                }
                yv = yn;
            }
            for (int i = 0; i < D; i++) err = std::max(err, (double)std::abs(hf(i, bb) - yv[i]));
        }
        printf("Inductive coupled scan (Euler)  D=%d B=%d T=%d\n", D, B, T);
        printf("  max abs err = %.3g -> %s\n", err, err < 1e-3 ? "PASS" : "FAIL");
        return err < 1e-3 ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
