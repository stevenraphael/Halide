// Viterbi decoding using an inductive Halide function.
//
// `prob` relies on a carve-out in Function::define_update(): a Func whose
// pure definition is exactly undef<>() may have a single update definition
// whose self-references recurse with shifted indices like an inductive
// function's pure definition would. `prev` is a plain RDom-based argmax and
// doesn't need this carve-out.

#include "Halide.h"
#include <cstdio>
#include <limits>
#include <vector>

using namespace Halide;

int main() {
  try {
    const int S = 4;   // number of hidden states
    const int M = 3;   // size of the emission alphabet
    const int T = 6;   // length of the observation sequence

    Var s("s"), t("t");

    Buffer<float> init(S);
    Buffer<float> trans(S, S);   // trans(r, s): P(state s | prev state r)
    Buffer<float> emit(S, M);    // emit(s, o): P(observation o | state s)
    Buffer<int> obs(T);

    for (int i = 0; i < S; i++) {
        init(i) = 1.0f / S;
    }
    for (int r = 0; r < S; r++) {
        float row_sum = 0;
        for (int c = 0; c < S; c++) {
            trans(r, c) = 1 + ((r + 1) * (c + 2)) % 5;
            row_sum += trans(r, c);
        }
        for (int c = 0; c < S; c++) {
            trans(r, c) /= row_sum;
        }
    }
    for (int st = 0; st < S; st++) {
        float row_sum = 0;
        for (int o = 0; o < M; o++) {
            emit(st, o) = 1 + ((st + 1) * (o + 3)) % 4;
            row_sum += emit(st, o);
        }
        for (int o = 0; o < M; o++) {
            emit(st, o) /= row_sum;
        }
    }
    for (int i = 0; i < T; i++) {
        obs(i) = (i * 2 + 1) % M;
    }

    RDom r(0, S, 0, S, "r");

    // obs(t) is data, so Halide can't otherwise prove it's a valid index
    // into emit's observation dimension; clamp for BoundsInference.
    Expr obs_t = Halide::Internal::promise_clamped(obs(Halide::Internal::promise_clamped(t, 0, T - 1)), 0, M - 1);
    Expr obs_0 = clamp(obs(0), 0, M - 1);

    // prob: inductive in t; pure def is just a placeholder, all real work
    // happens in the single update definition below.
    Func prob(Float(32), 2, "prob");
    prob(s, t) = (float)(0.0);

    // For t <= 0 this is the base case (recomputed redundantly per r,
    // harmless). For 0 < t < T it reduces over r, self-referencing
    // prob(r, t - 1) to take a running max of prob(t-1,r)*trans(r,s)*emit(s,obs(t)).
    // At t == T, prob(s, T) doubles as the terminal aggregate max_r
    // prob(r, T-1) (same value for every s), so prev below can find the
    // final state the same way it finds every backpointer.
    Expr prev_t = t - 1;
    // Candidate value contributed by state index "idx" (used both as prob's
    // running-max update with idx = r, and as cand2's formula below with an
    // ordinary index, since a pure definition can't be indexed by an RVar).
    auto normal_cand = [&](Expr idx, Expr idx2) {
        return likely(prob(idx, prev_t) * trans(idx2, idx) * emit(idx2, obs_t));
    };
    Expr cand = normal_cand(r.y, r.x);

    RDom r2(0, S, "r2");
    prob(r.x, t) = select(t <= 0,
                         init(r.x) * emit(r.x, obs_0),
                         likely(max(prob(r.x, t), cand)));

    // prev: NOT inductive, so its terminal-step (t == T) case can just be an
    // ordinary select instead of needing the arithmetic-blend trick prob
    // uses. The terminal row-max to compare against is an inline maximum()
    // reduction over a fresh RDom r3, only evaluated when is_terminal.
    RDom r3(0, S, "r3");
    Expr is_terminal = (t >= T);
    Expr terminal_cand = prob(r2, t - 1);  // at t == T: prob(r2, T - 1), the fully-resolved last row.
    Expr cand2 = select(is_terminal, terminal_cand, normal_cand(r2, s));
    Expr row_max = maximum(r3, prob(r3, t - 1));  // only meaningful/evaluated when is_terminal
    Expr compare_val = select(is_terminal, row_max, prob(s, t));

    Func prev(Int(32), 2, "prev");
    prev(s, t) = S;
    prev(s, t) = select(t <= 0, 0, likely(select(cand2 == compare_val, r2, likely(prev(s, t)))));

    Func path(Int(32), 1, "path");
    path(t) = undef<int>();
    path(T - 1) = prev(0, T);
    RDom rt(1, T - 1, "rt");
    // t = T - 1 - rt walks backwards from T - 2 down to 0. clamp() gives
    // BoundsInference a static bound on this data-dependent lookup.
    path(T - 1 - rt) = prev(clamp(path(T - rt), 0, S - 1), T - rt);

    // path indexes prev with a data-dependent state index, so bounds
    // inference can't infer prev's extent in s from its callers.
    prev.bound(s, 0, S).bound(t, 0, T + 1);
    prob.compute_root();
    prob.vectorize(s);
    prob.update().allow_race_conditions().vectorize(r.x);

    prev.vectorize(s).update().vectorize(s);
    prev.compute_root();
    path.compute_root();

    path.print_loop_nest();
    Buffer<int> path_result = path.realize({T});

    // Reference implementation in plain C++ for comparison.
    std::vector<std::vector<float>> ref_prob(T, std::vector<float>(S));
    std::vector<std::vector<int>> ref_prev(T, std::vector<int>(S, 0));
    for (int st = 0; st < S; st++) {
        ref_prob[0][st] = init(st) * emit(st, obs(0));
    }
    for (int tt = 1; tt < T; tt++) {
        for (int st = 0; st < S; st++) {
            float best = std::numeric_limits<float>::lowest();
            int best_r = 0;
            for (int rr = 0; rr < S; rr++) {
                float v = ref_prob[tt - 1][rr] * trans(st, rr) * emit(st, obs(tt));
                if (v > best) {
                    best = v;
                    best_r = rr;
                }
            }
            ref_prob[tt][st] = best;
            ref_prev[tt][st] = best_r;
        }
    }
    std::vector<int> ref_path(T);
    {
        int best_r = 0;
        float best = std::numeric_limits<float>::lowest();
        for (int st = 0; st < S; st++) {
            if (ref_prob[T - 1][st] > best) {
                best = ref_prob[T - 1][st];
                best_r = st;
            }
        }
        ref_path[T - 1] = best_r;
    }
    for (int tt = T - 2; tt >= 0; tt--) {
        ref_path[tt] = ref_prev[tt + 1][ref_path[tt + 1]];
    }

    bool ok = true;
    for (int tt = 0; tt < T; tt++) {
        if (path_result(tt) != ref_path[tt]) {
            printf("Mismatch at t=%d: halide=%d reference=%d\n",
                   tt, path_result(tt), ref_path[tt]);
            ok = false;
        }
    }

    if (!ok) {
        printf("Something went wrong!\n");
        return 1;
    }

    printf("Success!\n");
    return 0;
  } catch (const Halide::Error &e) {
      fprintf(stderr, "EXCEPTION: %s\n", e.what());
      return 1;
  }
}
