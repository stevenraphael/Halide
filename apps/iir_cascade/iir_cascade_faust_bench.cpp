// Benchmarks the Faust-generated cascade (iir_cascade_faust.cpp, compiled
// from iir_cascade.dsp) against the same T x S problem size used by
// test.cpp, so it can be compared directly against the Halide
// inductive/non-inductive numbers.
//
// Faust's dsp::compute() operates on one channel (one scanline) at a time,
// so we call it once per strip, matching how the Halide version treats s
// as the outer, parallel-over-strips dimension and t as the sequential
// recursive one.

#include <cmath>
#include <cstdio>
#include <vector>

#include "halide_benchmark.h"
#include "iir_cascade_faust.cpp"

using namespace Halide::Tools;

int main() {
    const int T = 1024;  // number of time steps
    const int S = 1024;  // number of strips

    std::vector<float> input(T * S);
    std::vector<float> output(T * S);

    for (int s = 0; s < S; s++) {
        for (int t = 0; t < T; t++) {
            input[s * T + t] = 0.5f * t + 10.0f * sinf(0.01f * t + 0.02f * s);
        }
    }

    mydsp dsp;
    dsp.init(44100);

    double t_faust = benchmark([&]() {
        for (int s = 0; s < S; s++) {
            FAUSTFLOAT *in_ptr = &input[s * T];
            FAUSTFLOAT *out_ptr = &output[s * T];
            FAUSTFLOAT *inputs[1] = {in_ptr};
            FAUSTFLOAT *outputs[1] = {out_ptr};
            dsp.instanceClear();
            dsp.compute(T, inputs, outputs);
        }
    });
    printf("Faust cascade time: %gms\n", t_faust * 1e3);

    return 0;
}
