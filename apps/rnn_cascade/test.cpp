#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

#include "HalideBuffer.h"
#include "HalideRuntime.h"

#include "rnn_cascade.h"
#include "rnn_cascade_noninductive.h"

#include "halide_benchmark.h"

using namespace Halide::Tools;

int main(int argc, char **argv) {
    const int C = 8;      // hidden / feature size
    const int T = 256;    // number of time steps
    const int S = 1024;   // number of strips (batch)
    const int N = 2;      // number of stacked layers

    Halide::Runtime::Buffer<float> input(C, T, S);
    Halide::Runtime::Buffer<float> Wx(C, C, N);
    Halide::Runtime::Buffer<float> Wh(C, C, N);
    Halide::Runtime::Buffer<float> bias(C, N);
    Halide::Runtime::Buffer<float> out_inductive(C, T, S);
    Halide::Runtime::Buffer<float> out_noninductive(C, T, S);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> small(-0.1f, 0.1f);

    input.for_each_element([&](int c, int t, int s) {
        input(c, t, s) = 0.1f * sinf(0.05f * t + 0.1f * c + 0.02f * s);
    });
    Wx.for_each_element([&](int ci, int co, int k) { Wx(ci, co, k) = small(rng); });
    Wh.for_each_element([&](int ci, int co, int k) { Wh(ci, co, k) = small(rng); });
    bias.for_each_element([&](int c, int k) { bias(c, k) = small(rng); });

    out_inductive(0, 0, 0) = 1.f;
    out_noninductive(0, 0, 0) = 2.f;

    double t_inductive = benchmark([&]() {
        rnn_cascade(input, Wx, Wh, bias, out_inductive);
        out_inductive.device_sync();
    });
    printf("inductive time:     %gms\n", t_inductive * 1e3);

    double t_noninductive = benchmark([&]() {
        rnn_cascade_noninductive(input, Wx, Wh, bias, out_noninductive);
        out_noninductive.device_sync();
    });
    printf("non-inductive time: %gms\n", t_noninductive * 1e3);

    float max_err = 0.f;
    out_inductive.for_each_element([&](int c, int t, int s) {
        max_err = std::max(max_err, std::abs(out_inductive(c, t, s) - out_noninductive(c, t, s)));
    });
    printf("max abs difference: %g\n", max_err);
    if (max_err > 1e-3f) {
        printf("Inductive and non-inductive outputs differ!\n");
        return 1;
    }

    printf("Success!\n");
    return 0;
}
