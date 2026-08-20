// Benchmark harness for the inductive-func Viterbi decoder in viterbi.cpp,
// scaled up to sizes where timing is meaningful, and dumping its inputs to a
// binary file so a Python script (bench_librosa.py) can run librosa's
// Viterbi decoder on the exact same data for a head-to-head comparison.
//
// Schedule is intentionally single-core (no .parallel anywhere -- the outer
// t loop is an inherently serial recurrence anyway) but vectorized across
// the state dimension s, matching the "vectorized, 1 core" comparison point.
//
// NOTE on the trans matrix layout: the Halide formula reads its `trans`
// buffer as trans(current, prev) (see normal_cand below), i.e. transposed
// relative to the standard row-stochastic transition-matrix convention
// (transition[from, to] = P(from->to), rows summing to 1). So this file
// builds the standard matrix as `trans_prob` and stores its *transpose* into
// the Halide buffer (trans(to, from) = trans_prob[from][to]), and dumps
// `trans_prob` itself -- untransposed -- to disk for bench_librosa.py, which
// matches librosa's convention directly with no further transpose needed.
#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    int S = argc > 1 ? atoi(argv[1]) : 64;    // number of hidden states
    int M = argc > 2 ? atoi(argv[2]) : 8;     // size of the emission alphabet
    int T = argc > 3 ? atoi(argv[3]) : 200000; // length of the observation sequence
    const char *data_path = argc > 4 ? argv[4] : "/tmp/viterbi_bench_data.bin";
    // "fold" (default): prob.compute_at(prev, t).store_root().fold_storage(t, 2)
    // "root": prob.compute_root() instead, materializing the whole prob array.
    std::string schedule = argc > 5 ? argv[5] : "fold";

    try {
        Var s("s"), t("t");

        Buffer<float> init(S);
        Buffer<float> trans(S, S);  // trans(r, c): P(state c | prev state r), row-stochastic.
        Buffer<float> emit(S, M);   // emit(s, o): P(observation o | state s), row-stochastic.
        Buffer<int> obs(T);

        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> unif(0.01f, 1.0f);

        {
            float sum = 0;
            for (int i = 0; i < S; i++) {
                init(i) = unif(rng);
                sum += init(i);
            }
            for (int i = 0; i < S; i++) init(i) /= sum;
        }
        // trans_prob[from][to] = P(from -> to), the standard row-stochastic
        // transition matrix (sums to 1 over "to" for each fixed "from").
        // The Halide formula reads its `trans` buffer as trans(current, prev)
        // -- i.e. transposed relative to that standard convention -- so the
        // buffer handed to the Halide pipeline is trans_prob's transpose:
        // trans(to, from) = trans_prob[from][to].
        std::vector<float> trans_prob(S * S);
        for (int from = 0; from < S; from++) {
            float row_sum = 0;
            for (int to = 0; to < S; to++) {
                trans_prob[from * S + to] = unif(rng);
                row_sum += trans_prob[from * S + to];
            }
            for (int to = 0; to < S; to++) trans_prob[from * S + to] /= row_sum;
        }
        for (int from = 0; from < S; from++) {
            for (int to = 0; to < S; to++) {
                trans(to, from) = trans_prob[from * S + to];
            }
        }
        for (int st = 0; st < S; st++) {
            float row_sum = 0;
            for (int o = 0; o < M; o++) {
                emit(st, o) = unif(rng);
                row_sum += emit(st, o);
            }
            for (int o = 0; o < M; o++) emit(st, o) /= row_sum;
        }
        std::uniform_int_distribution<int> obs_dist(0, M - 1);
        for (int i = 0; i < T; i++) obs(i) = obs_dist(rng);

        if (getenv("VITERBI_DEBUG_DUMP")) {
            for (int a = 0; a < std::min(S, 3); a++)
                for (int b = 0; b < std::min(S, 3); b++)
                    printf("trans(%d,%d)=%.6f trans_prob[%d*S+%d]=%.6f trans_prob[%d*S+%d]=%.6f\n",
                           a, b, trans(a, b), a, b, trans_prob[a * S + b], b, a, trans_prob[b * S + a]);
        }

        // Dump inputs (plain probabilities, not logs) for the Python/librosa
        // side -- librosa's public viterbi() API takes plain probabilities
        // and converts to log-space internally, matching the log-domain
        // Halide pipeline below.
        //
        // emit is built explicitly into a row-major (st * M + o) flat array
        // here rather than dumped via emit.data() directly: a Halide
        // Buffer<float>(S, M) stores dimension 0 (st) as the fastest-varying
        // stride, the opposite of numpy's default reshape(S, M) row-major
        // assumption, which silently produced a transposed matrix on the
        // Python side whenever S == M.
        std::vector<float> emit_row_major(S * M);
        for (int st = 0; st < S; st++)
            for (int o = 0; o < M; o++)
                emit_row_major[st * M + o] = emit(st, o);
        {
            FILE *f = fopen(data_path, "wb");
            int32_t header[3] = {S, M, T};
            fwrite(header, sizeof(int32_t), 3, f);
            fwrite(init.data(), sizeof(float), S, f);
            fwrite(trans_prob.data(), sizeof(float), S * S, f);
            fwrite(emit_row_major.data(), sizeof(float), S * M, f);
            fwrite(obs.data(), sizeof(int32_t), T, f);
            fclose(f);
        }

        // Log-domain versions of the inputs: plain-probability accumulation
        // underflows float32 within a few hundred steps of a long sequence
        // (each step multiplies by several factors < 1), so the recurrence
        // below sums logs and takes max instead of multiplying probabilities
        // and taking max -- the standard fix, and what librosa does
        // internally too.
        Buffer<float> log_init(S);
        Buffer<float> log_trans(S, S);
        Buffer<float> log_emit(S, M);
        for (int i = 0; i < S; i++) log_init(i) = std::log(init(i));
        for (int a = 0; a < S; a++)
            for (int b = 0; b < S; b++) log_trans(a, b) = std::log(trans(a, b));
        for (int st = 0; st < S; st++)
            for (int o = 0; o < M; o++) log_emit(st, o) = std::log(emit(st, o));

        RDom r(0, S, 0, S, "r");
        Expr obs_t = Halide::Internal::promise_clamped(obs(Halide::Internal::promise_clamped(t, 0, T - 1)), 0, M - 1);
        Expr obs_0 = clamp(obs(0), 0, M - 1);

        const float neg_inf = -std::numeric_limits<float>::infinity();

        Func prob(Float(32), 2, "prob");
        prob(s, t) = neg_inf;

        Expr prev_t = t - 1;
        auto normal_cand = [&](Expr idx, Expr idx2) {
            return likely(prob(idx, prev_t) + log_trans(idx2, idx) + log_emit(idx2, obs_t));
        };
        Expr cand = normal_cand(r.y, r.x);

        prob(r.x, t) = select(t <= 0,
                              log_init(r.x) + log_emit(r.x, obs_0),
                              likely(max(prob(r.x, t), cand)));

        RDom r2(0, S, "r2");
        RDom r3(0, S, "r3");
        Expr is_terminal = (t >= T);
        Expr terminal_cand = prob(r2, t - 1);
        Expr cand2 = select(is_terminal, terminal_cand, normal_cand(r2, s));
        Expr row_max = maximum(r3, prob(r3, t - 1));
        Expr compare_val = select(is_terminal, row_max, prob(s, t));

        Func prev(Int(32), 2, "prev");
        prev(s, t) = S;
        prev(s, t) = select(t <= 0, 0, likely(select(cand2 == compare_val, r2, likely(prev(s, t)))));

        Func path(Int(32), 1, "path");
        path(t) = undef<int>();
        path(T - 1) = prev(0, T);
        RDom rt(1, T - 1, "rt");
        path(T - 1 - rt) = prev(clamp(path(T - rt), 0, S - 1), T - rt);

        prev.bound(s, 0, S).bound(t, 0, T + 1);

        if (schedule == "root") {
            prob.compute_root();
        } else {
            prob.compute_at(prev, t).store_root().fold_storage(t, 2);
        }
        prob.vectorize(s);
        prob.update().allow_race_conditions().vectorize(r.x);

        prev.vectorize(s).update().vectorize(s);
        prev.compute_root();
        path.compute_root();

        // Warm-up (JIT compile happens on first realize call).
        Buffer<int> path_result(T);
        path.realize(path_result);

        const int trials = 5;
        double best_ms = 1e18;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            path.realize(path_result);
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best_ms) best_ms = ms;
        }

        printf("Halide (S=%d, M=%d, T=%d): best of %d = %.3f ms (%.2f Mstates/s)\n",
               S, M, T, trials, best_ms, (S * (double)T) / best_ms / 1000.0);

        // In-process reference check in the same log domain, using the log
        // buffers exactly as the Halide formula reads them (log_trans(cur,
        // prev)), to isolate whether a mismatch is a genuine Halide
        // schedule bug or just a Python-side data/indexing issue.
        {
            std::vector<std::vector<float>> ref_val(T, std::vector<float>(S));
            std::vector<std::vector<int>> ref_ptr(T, std::vector<int>(S, 0));
            for (int st = 0; st < S; st++) {
                ref_val[0][st] = log_init(st) + log_emit(st, obs(0));
            }
            for (int tt = 1; tt < T; tt++) {
                for (int st = 0; st < S; st++) {
                    float best = neg_inf;
                    int best_r = 0;
                    for (int rr = 0; rr < S; rr++) {
                        float v = ref_val[tt - 1][rr] + log_trans(st, rr) + log_emit(st, obs(tt));
                        if (v > best) {
                            best = v;
                            best_r = rr;
                        }
                    }
                    ref_val[tt][st] = best;
                    ref_ptr[tt][st] = best_r;
                }
            }
            std::vector<int> ref_path(T);
            {
                int best_r = 0;
                float best = neg_inf;
                for (int st = 0; st < S; st++) {
                    if (ref_val[T - 1][st] > best) {
                        best = ref_val[T - 1][st];
                        best_r = st;
                    }
                }
                ref_path[T - 1] = best_r;
            }
            for (int tt = T - 2; tt >= 0; tt--) {
                ref_path[tt] = ref_ptr[tt + 1][ref_path[tt + 1]];
            }
            if (getenv("VITERBI_DEBUG_DUMP")) {
                for (int tt = 0; tt < std::min(T, 3); tt++) {
                    printf("ref_val[%d]: ", tt);
                    for (int st = 0; st < S; st++) printf("%.6f ", ref_val[tt][st]);
                    printf("\n");
                }
                printf("ref_path: ");
                for (int tt = 0; tt < T; tt++) printf("%d ", ref_path[tt]);
                printf("\nhalide_path: ");
                for (int tt = 0; tt < T; tt++) printf("%d ", path_result(tt));
                printf("\n");
            }
            int mismatches = 0;
            for (int tt = 0; tt < T; tt++) {
                if (path_result(tt) != ref_path[tt]) mismatches++;
            }
            printf("In-process log-domain reference check: %d / %d mismatches\n", mismatches, T);
        }

        // Dump the resulting path so bench_librosa.py can compare it against
        // librosa's decode of the same data (both now in log-domain).
        {
            std::vector<int32_t> path_out(T);
            for (int i = 0; i < T; i++) path_out[i] = path_result(i);
            FILE *f = fopen((std::string(data_path) + ".path").c_str(), "wb");
            fwrite(path_out.data(), sizeof(int32_t), T, f);
            fclose(f);
        }

        return 0;
    } catch (const Halide::Error &e) {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
}
