// A cascade of N stacked IndRNN layers (Li et al., "Independently Recurrent
// Neural Network", CVPR 2018):
//
//   h_k(t, c, s) = tanh( sum_c' Wx[k](c, c') * src(t, c', s)
//                      + w_h[k](c) * h_k(t-1, c, s)
//                      + b[k](c) )
//
// where t is the (sequential) time axis, c is the hidden-unit axis, s is the
// batch/strip axis (parallel), and src is the input for layer 0 or the
// previous layer's hidden state for k > 0. Unlike a vanilla RNN, the
// recurrent connection is per-unit (diagonal), not a dense matrix: this is
// what makes it usable as an inductive function at all. Halide's inductive
// funcs require the self-recursive call to differ from the LHS in exactly
// one axis via a monotonic shift (t-1 here), with every other axis held
// identical; a dense recurrent matmul would require self-calls at varying c
// indices, which breaks that single-axis requirement and is rejected by
// Inductive.cpp ("not provably monotonically decreasing"). The input
// projection can still be a full dense matmul, since it calls src (a
// different, already-defined Func), not a self-call.
//
// This mirrors iir_cascade.cpp's structure (a cascade of N recurrent stages,
// each with an inductive and a non-inductive schedule) but with a dense
// input projection per layer, making it structurally closer to a real RNN
// cell than the per-channel independent IIR filter.

#include "Halide.h"

using namespace Halide;
using namespace Halide::BoundaryConditions;

class RNNCascade : public Generator<RNNCascade> {
public:
    Input<Buffer<float, 3>> input{"input"};    // (c, t, s): C input features, T steps, S strips
    Input<Buffer<float, 3>> Wx{"Wx"};          // (c_in, c_out, k): input-to-hidden weights per layer
    Input<Buffer<float, 3>> Wh{"Wh"};          // (c_in, c_out, k): hidden-to-hidden weights per layer
    Input<Buffer<float, 2>> bias{"bias"};      // (c, k): bias per layer
    GeneratorParam<int> N{"N", 2};             // number of stacked RNN layers
    GeneratorParam<int> C{"C", 8};             // hidden size (also input feature size)
    GeneratorParam<bool> inductive{"inductive", true};
    Output<Buffer<float, 3>> output{"output"};  // (c, t, s)

    void generate() {
        Var c("c"), t("t"), s("s");
        Func in_f("in");
        in_f(c, t, s) = BoundaryConditions::repeat_edge(input)(c, t, s);

        std::vector<Func> h(N);
        RDom rc(0, C, "rc");

        for (int k = 0; k < N; k++) {
            h[k] = Func(Float(32), "h" + std::to_string(k));
            Func src = (k == 0) ? in_f : h[k - 1];

            Expr b = bias(c, k);

            if (inductive) {
                // h[k] has no definition yet at this point, so the recurrent
                // term's self-call to h[k] can't be routed through sum()
                // (its helper Func would be calling an undefined Func).
                // Unroll the reduction manually instead.
                Expr input_term = sum(rc, Wx(rc, c, k) * src(rc, t, s), "input_term");
                Expr recur_term = 0.f;
                for (int rc_i = 0; rc_i < (int)C; rc_i++) {
                    recur_term += Wh(rc_i, c, k) * h[k](rc_i, t - 1, s);
                }
                h[k](c, t, s) = select(t <= 0,
                                        tanh(input_term + b),
                                        likely(tanh(input_term + recur_term + b)));
            } else {
                // sum()'s self-call to h[k] would "freeze" h[k] as soon as
                // it's used inside the helper Func's definition, which then
                // blocks adding h[k]'s own update definition afterwards.
                // Unroll the reduction manually to avoid routing the
                // self-call through a separate Func at all.
                h[k](c, t, s) = undef<float>();
                Expr input_term_0 = sum(rc, Wx(rc, c, k) * src(rc, 0, s), "input_term_0");
                h[k](c, 0, s) = tanh(input_term_0 + b);
                RDom rt(1, input.dim(1).extent() - 1, "rt");
                Expr input_term_rt = sum(rc, Wx(rc, c, k) * src(rc, rt, s), "input_term_rt");
                Expr recur_term_rt = 0.f;
                for (int rc_i = 0; rc_i < (int)C; rc_i++) {
                    recur_term_rt += Wh(rc_i, c, k) * h[k](rc_i, rt - 1, s);
                }
                h[k](c, rt, s) = tanh(input_term_rt + recur_term_rt + b);
            }
        }

        output(c, t, s) = undef<float>();
        RDom ro(0, input.dim(1).extent(), "ro");
        output(c, ro, s) = h[N - 1](c, ro, s);

        int VEC = 8;
        Var so("so"), si("si");
        if (get_target().has_feature(Target::CUDA)) {
            const int WARP = 32;
            output.update().split(s, so, si, WARP).gpu_blocks(so).gpu_lanes(si).reorder(c, ro, si, so);
            for (int k = 0; k < N; k++) {
                if (inductive) {
                    h[k].fold_storage(t, 2);
                    h[k].store_at(output, si).compute_at(output, ro).reorder_storage(c, t);
                } else {
                    h[k].compute_at(output, si).reorder_storage(c, t).store_in(MemoryType::Heap);
                    h[k].update(1).unroll(rc);
                }
            }
        } else {
            output.update()
                .split(s, so, si, VEC)
                .reorder(si, c, ro, so)
                .vectorize(si);
            for (int k = 0; k < N; k++) {
                if (inductive) {
                    h[k].reorder_storage(s, c, t).fold_storage(t, 2);
                    h[k].store_at(output, so).compute_at(output, ro).vectorize(s, VEC);
                } else {
                    h[k].compute_at(output, so).reorder_storage(s, c, t).vectorize(s, VEC).update().vectorize(s, VEC);
                    h[k].update(1).vectorize(s, VEC);
                }
            }

            output.dim(1).set_bounds(0, input.dim(1).extent());
        }
    }
};

HALIDE_REGISTER_GENERATOR(RNNCascade, rnn_cascade)
