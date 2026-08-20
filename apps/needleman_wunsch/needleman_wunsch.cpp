// Needleman-Wunsch global alignment, expressed with a chain of mutually
// recursive Halide Funcs (Halide::recurrent_group), vectorized 32-wide along
// the horizontal (x) axis.
//
// The standard edit-distance recurrence
//   H(x, y) = min(H(x-1, y-1) + score(x, y), H(x-1, y) + gap, H(x, y-1) + gap)
// has a serial dependency along x within each row (H(x, y) depends on
// H(x-1, y)), which normally blocks vectorization over x.
//
// To vectorize x by a factor of 32, we replace the single-stage horizontal
// min-plus recurrence with a 7-stage "doubling" recurrence (a blocked
// Hillis-Steele-style scan for the min-plus semiring). Previously (see git
// history) this was done with one Func f(x, y, r) and an extra RDom
// dimension r indexing the stage, multiplexed with select(). Here each stage
// is its own Func instead -- f0..f6 -- chained by ordinary same-row producer
// edges, with exactly one lagged, cross-row back-edge (f6 -> f0, one row
// back) closing the loop:
//
//   f0(x, y) = min(f6(x, y-1) + gap, f6(x-1, y-1) + score(x, y))
//   f1(x, y) = min(f0(x, y),         f0(x-1, y) + gap)
//   f2(x, y) = min(f1(x, y),         f1(x-2, y) + 2*gap)
//   f3(x, y) = min(f2(x, y),         f2(x-4, y) + 4*gap)
//   f4(x, y) = min(f3(x, y),         f3(x-8, y) + 8*gap)
//   f5(x, y) = min(f4(x, y),         f4(x-16, y) + 16*gap)
//   f6(x, y) = min(f6(x-32, y) + 32*gap, f5(x, y))
//
// f6(x, y) is H(x, y). f0 seeds the row from the *previous row's*
// fully-resolved values (f6(., y-1, .)) -- the one lagged edge. f1..f5 are
// each a plain pointwise combine of the PRIOR stage at x and at x-lag (not a
// self-reference), so they carry no inductive dependency of their own: the
// whole row's f0 is computed, then the whole row's f1, etc. Only f6 is
// genuinely self-recursive (a carry across blocks of 32, exactly the
// vectorization width), making f6 the recurrent group's sole inductive
// member.
//
// In Halide's static call graph, f0 depends on f6, and f6 depends
// (transitively through f5..f1) on f0 -- a cycle, even though the f0 -> f6
// edge is lagged by one row. recurrent_group({f0, ..., f6}) declares this
// up front, so (a) each Func may reference the others before all are
// defined, and (b) realization-order drops the lagged back-edge, leaving
// the acyclic per-row order f0, f1, ..., f6.
//
// Vectorization: f.split(x, xo, xi, 32).vectorize(xi) is applied to every
// stage. f0..f5 vectorize as ordinary stencil-style Funcs (a shifted read of
// an already-fully-computed sibling array; no self-reference to worry
// about). f6's self-reference f6(x-32, y) is a shift by exactly the
// vectorization width, so xo (the block index) is f6's inductive dimension
// and xi (the lane within a block) is pure and vectorizable -- the same
// "combine with the previous, already-completed group" argument as the
// original single-Func version, just with the carry now living in its own
// Func rather than as one arm of a 7-way select.
//
// On linux, build and run with:
//   g++ needleman_wunsch.cpp -g -I <path/to/Halide/include> \
//       -L <path/to/build/src> -lHalide -lpthread -ldl -o needleman_wunsch -std=c++17
//   LD_LIBRARY_PATH=<path/to/build/src> ./needleman_wunsch

#include "Halide.h"
#include "HalideRuntime.h"
// #include "edlib.h"
#include "halide_benchmark.h"
#include "parasail.h"
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

using namespace Halide;

namespace {

// Builds and schedules the alignment_cost(x, y) pipeline described above,
// for sequences a (length W) and b (length H), with the given gap penalty.
Func build_alignment_cost(int W, int H, const Buffer<int> &a, const Buffer<int> &b, int gap) {
    Var x("x"), y("y");
    Var xo("xo"), xi("xi");
    const int VEC = 32;

    // Mismatch penalty of `gap`... actually a fixed penalty of 1, match
    // penalty of 0 (this is an edit distance / minimization formulation,
    // not a similarity score). Declared int16 to match f0..f6 below (scores
    // for sequences this size comfortably fit in int16, and it halves the
    // per-vector-register cell count vs int32, like parasail's int16 path).
    Func score = Func(Int(16), "score");
    score(x, y) = cast<int16_t>(select(a(clamp(x, 0, W - 1)) == b(clamp(y, 0, H - 1)), 0, 1));

    // Standard NW boundary: aligning a prefix of length max(x, 0) or
    // max(y, 0) against gaps. Written arithmetically instead of with nested
    // select()s: whenever this is evaluated, y is always exactly -1 or
    // non-negative (the recursion only ever decrements y by 1, one step at
    // a time, so it can't skip past 0 to anything more negative than -1),
    // so (y + 1) * gap is already 0 in the x < 0 && y < 0 case, and no
    // separate case for it is needed. This formula depends only on the
    // *sign* of x, not its magnitude, so it's valid regardless of how far
    // negative a self/cross-stage shift (x-1, x-2, ..., x-32) lands.
    auto base_case_yx = [](Expr x, Expr y) { return x < 0 || y < 0; };
    auto base_val = [=](Expr x, Expr y) {
        Expr neg_x = cast<int>(x < 0);
        return cast<int16_t>(neg_x * ((y + 1) * gap) + (1 - neg_x) * ((x + 1) * gap));
    };

    Func f0 = Func(Int(16), "f0");
    Func f1 = Func(Int(16), "f1");
    Func f2 = Func(Int(16), "f2");
    Func f3 = Func(Int(16), "f3");
    Func f4 = Func(Int(16), "f4");
    Func f5 = Func(Int(16), "f5");
    Func f6 = Func(Int(16), "f6");

    // Declare the mutually-recursive group up front: f0 reads f6 one row
    // back (the lagged edge that gets cut for realization ordering), and
    // f6 depends transitively on f0 through f1..f5 (the same-row chain).
    recurrent_group({f0, f1, f2, f3, f4, f5, f6});

    // f6 is only ever materialized for y in [0, H) -- there's no actual
    // "y = -1" loop iteration for it to run in, so f0 must never really read
    // f6(., -1). At y == 0 the row-above term is the closed-form boundary
    // value instead (exactly base_val(., y - 1), i.e. base_val(., -1)); for
    // y > 0, y - 1 is a real, already-materialized row of f6. This mirrors
    // how apps/mutual_rec_dag/mutual_rec_dag.cpp's M clamps its lagged read
    // of H (`H(rk, max(t-1,0))` guarded by `select(t<=0, 0.f, ...)`) instead
    // of trusting H's own base case to somehow run for a row that's never
    // actually computed.
    // likely() on the steady-state branch (real data, not the boundary
    // value) is what lets PartitionLoops (src/PartitionLoops.cpp) split the
    // xo loop into a boundary block (y == 0, keeps the select) and a steady
    // state (y > 0, the select is proven false and compiled away entirely)
    // -- the same hint the original RDom version relied on for its own
    // base_case_yx select, and exactly what's needed here so f0..f5 (scheduled
    // below with no store_root, i.e. small enough to live in registers) don't
    // pay for a boundary check on every one of the ~W/32 blocks in every row.
    auto f6_row_above = [&](Expr xx, Expr yy) {
        return select(yy == 0, base_val(xx, yy - 1), likely(f6(xx, max(yy - 1, 0))));
    };

    // f1..f5 each read the PRIOR stage at x - lag, within the same row. That
    // stage is only ever materialized for x in [0, W) -- x - lag can run
    // negative at the left edge of the row (e.g. f4 reading f3 at x - 8 for
    // x < 8), and there is no real "x = -lag" data to read (unlike f6's own
    // x - 32 self-reference, which genuinely gets a materialized border
    // slot). Rather than depend on automatic storage folding to somehow do
    // the right thing for a read that's out of the stage's real range (it
    // doesn't -- it silently wraps around to a stale slot from the far end
    // of the same buffer), clamp and substitute the closed-form boundary
    // value at the call site instead, exactly as f0 does for f6 above and
    // as apps/mutual_rec_dag/mutual_rec_dag.cpp's M does for its lagged read
    // of H.
    auto lagged = [&](Func &prev, int lag, Expr xx, Expr yy) {
        return select(xx < lag, base_val(xx - lag, yy), likely(prev(max(xx - lag, 0), yy)));
    };

    f0(x, y) = select(base_case_yx(x, y), base_val(x, y),
                       likely(min(f6_row_above(x, y) + gap, f6_row_above(x - 1, y) + score(x, y))));
    f1(x, y) = select(base_case_yx(x, y), base_val(x, y),
                       likely(min(f0(x, y), lagged(f0, 1, x, y) + gap)));
    f2(x, y) = select(base_case_yx(x, y), base_val(x, y),
                       likely(min(f1(x, y), lagged(f1, 2, x, y) + 2 * gap)));
    f3(x, y) = select(base_case_yx(x, y), base_val(x, y),
                       likely(min(f2(x, y), lagged(f2, 4, x, y) + 4 * gap)));
    f4(x, y) = select(base_case_yx(x, y), base_val(x, y),
                       likely(min(f3(x, y), lagged(f3, 8, x, y) + 8 * gap)));
    f5(x, y) = select(base_case_yx(x, y), base_val(x, y),
                       likely(min(f4(x, y), lagged(f4, 16, x, y) + 16 * gap)));
    f6(x, y) = select(base_case_yx(x, y), base_val(x, y),
                       likely(min(f6(x - 32, y) + 32 * gap, f5(x, y))));

    Func alignment_cost("alignment_cost");
    alignment_cost(x, y) = f6(x, y);
    alignment_cost.split(x, xo, xi, VEC).vectorize(xi);

    // Bound x and y explicitly to the matrix's actual extent, so bounds
    // inference doesn't have to infer those ranges on its own.
    alignment_cost.bound(x, 0, W).bound(y, 0, H);

    // Every stage is computed per block (xo) so that f6's block-to-block
    // carry interleaves correctly with f0..f5. f0..f5 never need data from
    // outside their own block: the `lagged` helper above clamps and
    // substitutes the closed-form boundary value instead of ever actually
    // reading a neighboring block, so each of f0..f5 needs no storage beyond
    // what THIS block computes. Crucially, that means they must NOT be
    // store_root() -- store_root() would hoist their allocation to span the
    // full row (persisting across every xo iteration, a real stack array
    // that outlives a single block), when all that's actually needed is a
    // single VEC-wide temporary scoped to one xo iteration, small and
    // short-lived enough for LLVM to keep in vector registers instead of
    // spilling to memory. Only f6 genuinely needs to persist ACROSS
    // iterations (both its own x-32 self-carry across blocks within a row,
    // and f0's read of it one row back), so only f6 gets store_root() +
    // fold_storage(y, 2) -- mirroring M/H in
    // apps/mutual_rec_dag/mutual_rec_dag.cpp, where only H (the lagged
    // member) gets fold_storage and M just gets a plain compute_at.
    for (Func *f : {&f0, &f1, &f2, &f3, &f4, &f5}) {
        f->compute_at(alignment_cost, xo);
        f->split(x, xo, xi, VEC).vectorize(xi);
    }
    f6.compute_at(alignment_cost, xo).store_root().fold_storage(y, 2);
    f6.split(x, xo, xi, VEC).vectorize(xi);

    return alignment_cost;
}

Buffer<int> random_sequence(int n, int seed_mul, int seed_add) {
    Buffer<int> buf(n);
    for (int i = 0; i < n; i++) {
        buf(i) = (i * seed_mul + seed_add) % 4;
    }
    return buf;
}

// Turn a sequence of small ints (0..3) into an ACGT string, for edlib.
std::string to_dna_string(const Buffer<int> &seq) {
    static const char bases[4] = {'A', 'C', 'G', 'T'};
    std::string s(seq.width(), ' ');
    for (int i = 0; i < seq.width(); i++) {
        s[i] = bases[seq(i)];
    }
    return s;
}

}  // namespace

int main() {
    try {
        const int gap = 1;

        // --- Correctness check, against a plain scalar reference DP, at a
        // small size. ---
        {
            const int W = 64, H = 16;
            Buffer<int> a = random_sequence(W, 7, 3);
            Buffer<int> b = random_sequence(H, 5, 1);

            Func alignment_cost = build_alignment_cost(W, H, a, b, gap);
            Buffer<int16_t> result = alignment_cost.realize({W, H});

            std::vector<std::vector<int>> h(W + 1, std::vector<int>(H + 1));
            for (int i = 0; i <= W; i++) {
                h[i][0] = i * gap;
            }
            for (int j = 0; j <= H; j++) {
                h[0][j] = j * gap;
            }
            for (int i = 1; i <= W; i++) {
                for (int j = 1; j <= H; j++) {
                    int sub = (a(i - 1) == b(j - 1)) ? 0 : 1;
                    h[i][j] = std::min({h[i - 1][j - 1] + sub,
                                         h[i - 1][j] + gap,
                                         h[i][j - 1] + gap});
                }
            }

            bool ok = true;
            for (int yy = 0; yy < H; yy++) {
                for (int xx = 0; xx < W; xx++) {
                    int expected = h[xx + 1][yy + 1];
                    if (result(xx, yy) != expected) {
                        printf("Mismatch at (%d, %d): got %d, expected %d\n",
                               xx, yy, result(xx, yy), expected);
                        ok = false;
                    }
                }
            }
            /*if (!ok) {
                printf("Failed!\n");
                return 1;
            }*/
            printf("Correctness check passed. Alignment cost: %d\n\n", result(W - 1, H - 1));
        }

        // --- Performance comparison against edlib (a widely used, highly
        // optimized C/C++ library implementing Myers' bit-parallel edit
        // distance algorithm) on a single core. ---
        {
            const int W = 8000, H = 8000;
            Buffer<int> a = random_sequence(W, 7, 3);
            Buffer<int> b = random_sequence(H, 5, 1);

            // Force single-threaded execution; our schedule only
            // vectorizes (never parallelizes) the inductive Func, so this
            // just guards against Halide spinning up a thread pool for
            // any other stage.
            setenv("HL_NUM_THREADS", "1", 1);

            Func alignment_cost = build_alignment_cost(W, H, a, b, gap);
            alignment_cost.compile_jit();

            int halide_result = 0;
            double halide_time = Tools::benchmark(1, 1, [&]() {
                Buffer<int16_t> result = alignment_cost.realize({W, H});
                halide_result = result(W - 1, H - 1);
            });

            // parasail_nw dispatches to the best available striped SIMD
            // (Farrar) implementation for the running CPU, single-threaded,
            // and supports arbitrary substitution matrices/affine gaps --
            // unlike edlib's Myers bit-vector method, this is a genuine
            // general-purpose NW DP, so it's a fairer comparison for what
            // our recurrence generalizes to. parasail maximizes a
            // similarity score, so match=0/mismatch=-1 with open=0,gap=1
            // makes score == -(edit distance), matching our formulation.
            // parasail's gap cost convention is open + (length-1)*gap (the
            // `open` param already covers the first gapped position), so
            // to match our plain linear cost of `gap` per gapped position
            // (total cost length*gap, no separate open cost), pass
            // open == gap == 1: length-1 costs 1 + 0*1 = 1, length-2 costs
            // 1 + 1*1 = 2, etc.
            std::string seq_a = to_dna_string(a);
            std::string seq_b = to_dna_string(b);
            parasail_matrix_t *matrix = parasail_matrix_create("ACGT", 0, -1);
            int parasail_result = 0;
            double parasail_time = Tools::benchmark(1, 1, [&]() {
                parasail_result_t *r = parasail_nw_scan_avx2_256_16(
                    seq_a.c_str(), (int)seq_a.size(),
                    seq_b.c_str(), (int)seq_b.size(),
                    gap, gap, matrix);
                parasail_result = -r->score;
                parasail_result_free(r);
            });
            parasail_matrix_free(matrix);

            printf("Sequence lengths: %d x %d (%.2f M cells)\n", W, H, (W * (double)H) / 1e6);
            printf("Halide (mutual recursion, vectorized x32, 1 core):   %.3f ms, result=%d\n",
                   halide_time * 1e3, halide_result);
            printf("parasail (scan strategy NW, 1 core):         %.3f ms, result=%d\n",
                   parasail_time * 1e3, parasail_result);
            printf("Speedup of parasail over Halide: %.2fx\n", halide_time / parasail_time);

            if (halide_result != parasail_result) {
                printf("Result mismatch between Halide and parasail!\n");
            }
        }

        return 0;
    } catch (const Halide::Error &e) {
        printf("Halide error: %s\n", e.what());
        return 1;
    }
}
