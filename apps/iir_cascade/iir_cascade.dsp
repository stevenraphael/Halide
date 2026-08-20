// Faust equivalent of the Halide iir_cascade app: a cascade of N first-order
// forward IIR filters with a tanh nonlinearity applied before each feedback
// stage, matching the recursion used in iir_cascade_generator.cpp:
//
//   filt[k](t) = (1-weight)*filt[k](t-1) + weight*tanh(gain*src(t))
//
// (Faust's `~` feedback implicitly zero-initializes the delayed sample, so
// the t=0 boundary behaves slightly differently than the hand-written
// Halide version, which skips the tanh on the very first sample. This does
// not affect steady-state performance, which is what this file is for.)

import("stdfaust.lib");

N = 4;
weight = 0.3;
gain = 3.0;

stage(x) = x : *(gain) : ma.tanh : *(weight) : + ~ *(1.0 - weight);

process = seq(k, N, stage) : *(gain) : ma.tanh;
