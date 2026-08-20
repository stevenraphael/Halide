// RNN recurrence as ONE inductive Func with two update stages -- no mutual
// recursion, no second Func, no cycle in the DAG:
//
//   h(d, t) = undef                              (pure placeholder)
//   h(d, t) += A(r, d) * h(r, t - 1)             (update 0: matmul; the
//                                                 self-ref h(r,t-1) reads
//                                                 t-1's FINAL, post-tanh
//                                                 value)
//   h(d, t)  = tanh(h(d, t) + x(d, t))           (update 1: in-place
//                                                 activation)
//
// The base case (t <= 0, where h_{-1} = 0 so h_0 = tanh(x_0)) is folded
// into each update via select on t.

#include "Halide.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace Halide;

int main() {
    const int N = 8, T = 20;
    Var d("d"), t("t");
    // Output index carried as RVars (rd), matmul reduction as rk -- so the
    // d-position on every update LHS is an RVar, keeping d a plain reduction
    // dimension (NOT inductive). Only t is inductive.
    RDom rdk(0, N, 0, N, "rdk");   // rdk.x = output d, rdk.y = matmul reduce
    RVar rd = rdk.x, rk = rdk.y;
    RDom rd1(0, N, "rd1");          // output d for the activation update
    ImageParam A(Float(32), 2, "A");
    ImageParam x_p(Float(32), 2, "x_p");

    Func h(Float(32), 2, "h");
    try {
        h(d, t) = 0.f;
        // Update 0: matmul accumulate over rk (once per rd). At t<=0 adds 0
        // (h_{-1}=0 -> stays 0); else accumulates A @ h(t-1). The self-ref
        // h(rk, t-1) reads t-1 after ALL its update stages (incl. the tanh
        // below) -> the post-activation hidden state.
        h(rd, t) = select(t <= 0, 0.f, likely(h(rd, t) + A(rk, rd) * h(rk, t - 1)));
        // Update 1: add input and apply activation, in place (per rd).
        // Wrapped in a never-true select so the accumulator self-ref
        // h(rd1,t) lives inside a select value, satisfying the inductive
        // machinery's "self-reference must be inside a select" structural
        // check (the t<=-1 base branch never fires for t>=0).
        h(rd1, t) = select(t <= -1, 0.f,
                           likely(tanh(h(rd1, t) + x_p(rd1, t))));
    } catch (const Halide::Error &e) {
        fprintf(stderr, "CONSTRUCTION ERROR: %s\n", e.what());
        return 1;
    }

    // Inductive funcs can't be pipeline outputs; wrap in a plain Func.
    Func out("out");
    try {
        out(d, t) = h(d, t);
        out.compute_root();
        // Compute h one t-column at a time inside out's t loop, with storage
        // rolled to a 2-deep window (t and t-1) via sliding-window folding --
        // h only ever reads h(*, t-1), so it never needs the full T history.
        h.compute_at(out, t).store_root().fold_storage(t, 2);
    } catch (const Halide::Error &e) {
        fprintf(stderr, "CONSTRUCTION ERROR: %s\n", e.what());
        return 1;
    }

    Buffer<float> A_buf(N, N), x_buf(N, T);
    std::vector<float> A_v(N * N), x_v(N * T);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float v = 0.1f * ((i * 7 + j * 3) % 5 - 2);
            A_buf(j, i) = v;
            A_v[j + i * N] = v;
        }
    for (int tt = 0; tt < T; tt++)
        for (int dd = 0; dd < N; dd++) {
            float v = 0.05f * ((dd * 3 + tt) % 7 - 3);
            x_buf(dd, tt) = v;
            x_v[dd + tt * N] = v;
        }
    A.set(A_buf);
    x_p.set(x_buf);

    Buffer<float> result;
    try {
        result = out.realize({N, T});
    } catch (const Halide::Error &e) {
        fprintf(stderr, "REALIZE ERROR: %s\n", e.what());
        return 1;
    }

    // Reference: h_t = tanh(A @ h_{t-1} + x_t), h_{-1} = 0.
    std::vector<float> ref(N, 0.f);
    std::vector<float> ref_out(N * T);
    for (int tt = 0; tt < T; tt++) {
        std::vector<float> next(N);
        for (int dd = 0; dd < N; dd++) {
            float acc = 0.f;
            for (int rr = 0; rr < N; rr++) acc += A_v[rr + dd * N] * ref[rr];
            next[dd] = std::tanh(acc + x_v[dd + tt * N]);
        }
        ref = next;
        for (int dd = 0; dd < N; dd++) ref_out[dd + tt * N] = ref[dd];
    }

    // Alternate reference B: recurrence reads the PRE-activation (post-
    // matmul, pre-tanh) previous state, to detect if h(rk,t-1) is reading
    // the wrong stage.
    std::vector<float> refB_prev_post(N, 0.f), refB_out(N * T);
    for (int tt = 0; tt < T; tt++) {
        std::vector<float> pre(N), post(N);
        for (int dd = 0; dd < N; dd++) {
            float acc = 0.f;
            for (int rr = 0; rr < N; rr++) acc += A_v[rr + dd * N] * refB_prev_post[rr];
            pre[dd] = acc;
            post[dd] = std::tanh(acc + x_v[dd + tt * N]);
        }
        // Variant B: next step reads pre (pre-activation) instead of post.
        refB_prev_post = pre;
        for (int dd = 0; dd < N; dd++) refB_out[dd + tt * N] = post[dd];
    }

    float max_err = 0, max_errB = 0;
    for (int tt = 0; tt < T; tt++)
        for (int dd = 0; dd < N; dd++) {
            max_err = std::max(max_err, std::abs(result(dd, tt) - ref_out[dd + tt * N]));
            max_errB = std::max(max_errB, std::abs(result(dd, tt) - refB_out[dd + tt * N]));
        }
    printf("post-activation ref err=%.6e   pre-activation ref err=%.6e\n", max_err, max_errB);

    printf("max_err=%.6e  %s\n", max_err, max_err < 1e-4f ? "PASS" : "FAIL");
    return max_err < 1e-4f ? 0 : 1;
}
