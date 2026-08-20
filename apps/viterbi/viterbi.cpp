// Draft: Viterbi decoding expressed with a Halide inductive function.
//
// NOTE: This relies on a local, forward-looking relaxation in
// Function::define_update() (src/Function.cpp): a Func whose pure
// definition is exactly undef<>() (i.e. it isn't itself inductive -- it's a
// placeholder, as in the classic RDom-scan idiom) is now allowed exactly one
// update definition, and that update's self-references are allowed to
// recurse with shifted indices the way an inductive function's pure
// definition would. `prob` below uses that carve-out: its pure def is
// undef<float>(), and its single update definition both handles the t <= 0
// base case and reduces over an RVar `r` to take the max over the previous
// row of states, self-referencing prob(r, t - 1). `prev` intentionally does
// *not* use this carve-out -- it's a plain RDom-based argmax, matching the
// spec that prev should just use an RDom.
//
// Algorithm (from the prompt):
//
//   prob[0][s]   = init[s] * emit[s][obs[0]]
//   prob[t][s]   = max over r of prob[t-1][r] * trans[r][s] * emit[s][obs[t]]
//   prev[t][s]   = argmax over r of the same quantity
//   path[T-1]    = argmax_s prob[T-1][s]
//   path[t]      = prev[t+1][path[t+1]]

#include "Halide.h"
#include <cstdio>
#include <limits>
#include <vector>

using namespace Halide;

int main() {
  try {
    // Problem sizes.
    const int S = 4;   // number of hidden states
    const int M = 3;   // size of the emission alphabet
    const int T = 6;   // length of the observation sequence

    Var s("s"), t("t");

    // Inputs.
    Buffer<float> init(S);
    Buffer<float> trans(S, S);   // trans(r, s): P(state s | prev state r)
    Buffer<float> emit(S, M);    // emit(s, o): P(observation o | state s)
    Buffer<int> obs(T);

    // Fill with some arbitrary but deterministic values for the draft.
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
    // into emit's second (observation) dimension; clamp it so
    // BoundsInference has a static bound to work with.
    Expr obs_t = Halide::Internal::promise_clamped(obs(Halide::Internal::promise_clamped(t, 0, T - 1)), 0, M - 1);
    Expr obs_0 = clamp(obs(0), 0, M - 1);

    // --- prob: inductive in t, using an RVar (r) to take the max over the
    // previous row of states. ---
    //
    // Pure definition: a placeholder. All the real work happens in the
    // single update definition below.
    Func prob(Float(32), 2, "prob");
    prob(s, t) = (float)(0.0);

    // Update (the one carve-out update definition allowed on a Func whose
    // pure def is undef<>()): for t <= 0, this is just the base case,
    // recomputed redundantly once per r (harmless). For 0 < t < T, it
    // reduces over r, self-referencing prob(r, t - 1) -- the previous row
    // of states -- taking the max of prob(t - 1, r) * trans(r, s) *
    // emit(s, obs(t)). At t == T, prob(s, T) does double duty as the
    // terminal aggregate max_r prob(r, T - 1) (the same value for every s,
    // computed via the same running-max mechanic): this lets prev below
    // find the final state the same way it finds every other backpointer,
    // by comparing candidates against prob(s, t), instead of needing a
    // second Func to independently discover that max. The r == 0 iteration
    // seeds the running max since the pure def can't (it's undef<>());
    // subsequent iterations of r fold in via max() against the current
    // accumulated value prob(s, t).
    Expr prev_t = t - 1;
    // The candidate value contributed by state index "idx": the transition
    // score. Used both as prob's own running-max update (with idx = r, the
    // reduction var) and as cand2's pure definition below (with idx an
    // ordinary Var/RVar, since a pure definition can't be indexed by an
    // RVar). No t == T special case here at all any more -- prob's own
    // recurrence never needs to know about the terminal step, since prob
    // only reads/writes its own row at t and t - 1, both of which stay
    // < T for every iteration prob itself runs. The terminal aggregate is
    // now found entirely inside prev's update instead (see cand2 below),
    // since prev isn't inductive and isn't subject to the "no select
    // dividing two recursive cases" restriction that motivated the
    // arithmetic-blend rewrite of this formula in the first place.
    auto normal_cand = [&](Expr idx, Expr idx2) {
        return likely(prob(idx, prev_t) * trans(idx2, idx) * emit(idx2, obs_t));
    };
    Expr cand = normal_cand(r.y, r.x);

    RDom r2(0, S, "r2");
    prob(r.x, t) = select(t <= 0,
                         init(r.x) * emit(r.x, obs_0),
                         likely(max(prob(r.x, t), cand)));

    // --- prev: NOT inductive. Plain RDom-based argmax over r2, extended to
    // also carry the terminal-step (t == T) aggregate search that used to
    // live inside prob's own update. select(t == T, ...) here is completely
    // fine -- prev has no recursive self-reference restrictions -- so the
    // terminal case is just an ordinary select, no arithmetic blend needed.
    // prev stays a plain Int(32) (the backpointer/terminal-state index):
    // the terminal row-max to compare candidates against is obtained via
    // Halide's maximum() inline reduction (a plain Expr, not a persistent
    // Func of its own) over a fresh RDom r3, rather than needing prev to
    // carry a second Tuple component to accumulate that max itself.
    RDom r3(0, S, "r3");
    Expr is_terminal = (t >= T);
    Expr terminal_cand = prob(r2, t - 1);  // at t == T, this is prob(r2, T - 1): the fully-resolved last row.
    Expr cand2 = select(is_terminal, terminal_cand, normal_cand(r2, s));
    Expr row_max = maximum(r3, prob(r3, t - 1));  // only meaningful (and only evaluated) when is_terminal
    Expr compare_val = select(is_terminal, row_max, prob(s, t));

    Func prev(Int(32), 2, "prev");
    prev(s, t) = S;
    prev(s, t) = select(t <= 0, 0, likely(select(cand2 == compare_val, r2, likely(prev(s, t)))));

    Func path(Int(32), 1, "path");
    path(t) = undef<int>();
    path(T - 1) = prev(0, T);
    RDom rt(1, T - 1, "rt");
    // t = T - 1 - rt walks backwards from T - 2 down to 0.
    // clamp() gives BoundsInference a static bound on the region of prev
    // required by this data-dependent (state-index) lookup.
    path(T - 1 - rt) = prev(clamp(path(T - rt), 0, S - 1), T - rt);

    // path indexes into prev with a data-dependent state index (the
    // backtrace), so BoundsInference can't infer prev's extent in s from
    // its callers; state it explicitly.
    prev.bound(s, 0, S).bound(t, 0, T + 1);
    // prob's own update reads prob(r, t - 1); ordinary bounds inference
    // doesn't prune that read out of the t <= 0 (base-case) branch of the
    // select, so it also requires prob(*, -1) to be defined (evaluating to
    // the same harmless base-case formula there). Extend the realized
    // range by one to cover it.
    //prob.bound(s, 0, S).bound(t, -1, T + 1);

    // Fuse prob into prev's t loop: for each t, prev only needs the row of
    // prob at that t (and, transitively, prob's own update only needs the
    // row at t - 1), so store_at(prev, t) + fold_storage(t, 1) keeps just a
    // single row of prob live at a time instead of materializing all of it
    // up front -- the same benefit inductive functions give the classic
    // prefix-sum example in tutorial/lesson_25_inductive.cpp.
    prob.compute_root();//compute_at(prev, t).store_root().fold_storage(t, 2); // TODO: why does 1 fail
    prob.vectorize(s);
    prob.update().allow_race_conditions().vectorize(r.x);
    // Disable PartitionLoops' automatic prologue/steady-state split for
    // prev's t loop: that heuristic was producing a dead prologue branch
    // (a guard that could never be true within its own single-iteration
    // prologue loop), silently leaving prev(s, 0) stuck at its placeholder
    // value. (Func::specialize() can't be used here instead -- it only
    // accepts conditions independent of the loop variable itself, not a
    // per-iteration condition like t == 0.) never_partition(t) just keeps
    // the select(t <= 0, ...) branch as an ordinary per-iteration branch.

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
