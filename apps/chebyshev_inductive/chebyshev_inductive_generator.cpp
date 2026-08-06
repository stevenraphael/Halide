// Chebyshev semi-iteration for an SPD system A x = b.
// Contains a version that uses inductive functions, and a version that does not.
//
// The Chebyshev iterate obeys a single-sequence three-term recurrence
//
//     x_{k+1} = (1 + w_k) x_k - w_k x_{k-1} + a_k (b - A x_k)
//
// whose coefficients a_k, w_k depend only on the spectral bounds of A (there are no
// inner products). So a step is one mat-vec plus a bounded-lag, same-component
// combination -- exactly the shape an inductive Func can express and fold.
//
// The recurrence is a *mat-vec* recurrence: column k reads the previous iterate at
// *every* component, so the self-reference puts a summed-over index where the
// output component sits. The inductive version keeps the whole iteration in one
// pipeline and folds its storage to O(1) columns. The non-inductive version
// expresses the SAME recurrence as a single Func scanned with a multi-dimensional
// reduction domain (no unrolled vector<Func>), folded by hand into a 3-column ring
// (index k % 3) so that both versions use O(1)-column storage and the comparison is
// fair.

#include "Halide.h"

using namespace Halide;

class ChebyshevInductive : public Generator<ChebyshevInductive> {
public:
    Input<Buffer<double, 2>> A{"A"};          // n x n SPD matrix (column-major)
    Input<Buffer<double, 1>> b{"b"};          // right-hand side, length n
    Input<Buffer<double, 1>> alpha{"alpha"};  // per-iteration coefficient a_k, length M
    Input<Buffer<double, 1>> omega{"omega"};  // per-iteration coefficient w_k, length M

    GeneratorParam<int> M{"M", 60};  // number of Chebyshev iterations
    GeneratorParam<bool> inductive{"inductive", true};

    Output<Buffer<double, 1>> x{"x"};  // solution iterate x_M, length n

    void generate() {
        Var t("t"), k("k");
        Expr n = A.dim(0).extent();

        if (inductive) {
            // ---- inductive: the whole iteration is ONE folded pipeline ----
            // The mat-vec couples every output component to *every* component of the
            // previous column, so the self-reference X(j, k-1) puts a component index
            // where the output component sits. That index cannot be the inductive
            // (monotonically-decreasing) one -- so the ONLY inductive dimension is the
            // iteration k. The two coupled components (output i and mat-vec j) are BOTH
            // reduction variables of a 2-D RDom, nested INSIDE the inductive k. Only k
            // recurses (with a bounded lag of 2), so only k needs to fold.
            Func X(Float(64), "X");
            // Pure (non-inductive) definition: base case (k <= 0) is x0 = 0; the
            // accumulator for the update starts at 0.
            X(t, k) = cast<double>(0);

            // r.x = i (output component), r.y = j (mat-vec component). Both are RVars
            // and both live inside the inductive variable k.
            RDom r(0, n, 0, n, "r");
            Expr km1 = max(0, k - 1), km2 = max(0, k - 2);
            Expr once = (Expr(1.0) + omega(km1)) * X(r.x, km1) - omega(km1) * X(r.x, km2) +
                        alpha(km1) * b(r.x);
            // One update. Every self-reference is monotone in k (the accumulator
            // X(r.x, k), the lagged X(r.x, k-1)/X(r.x, k-2), and the mat-vec column
            // X(r.y, k-1)); dimension 0 is a reduction dimension, not inductive, so the
            // component index r.y there is fine. Once-per-column terms are gated by
            // r.y == 0; the mat-vec of the previous column is accumulated over r.y.
            X(r.x, k) = select(k <= 0, cast<double>(0),
                               X(r.x, k) + cast<double>(r.y == 0) * once -
                                   alpha(km1) * A(r.x, r.y) * X(r.y, km1));

            // Endpoint-extract consumer: keep only column M (= x_M), so the OUTPUT is
            // O(n). The iteration index rk is the OUTERMOST loop (the mat-vec couples
            // all components within a column), which is also the sweep that lets X fold.
            RDom rk(0, (int)M + 1, "rk");
            x(t) = cast<double>(0);
            x(t) += select(rk == (int)M, X(t, rk), cast<double>(0));
            x.update(0).reorder(t, rk);
            X.compute_at(x, rk).store_root().fold_storage(k, 3);
            // r.x (output component) is innermost -> vectorizable; distinct r.x write
            // distinct rows of column k, so the race is benign (as in the non-inductive
            // version).
            X.update(0).allow_race_conditions().vectorize(r.x, 4);
        } else {
            // ---- non-inductive: ONE Func, scanned with a multi-dimensional RDom,
            //      folded by hand into a 3-column ring ----
            // Same 3-D reduction as before -- output component (i = r.x), mat-vec
            // component (j = r.y), iteration (k = r.z) -- but instead of materialising
            // the whole n x (M+1) history we keep only three live columns and index the
            // iteration dimension modulo 3, mirroring the inductive fold_storage(k, 3)
            // by hand. This makes the comparison fair on storage: both versions use O(n)
            // working memory, so any timing difference is scheduling, not I/O volume.
            Func X(Float(64), "Xni");
            X(t, k) = cast<double>(0);  // pure def: all three ring columns start at 0

            // r.x = i, r.y = j, r.z = k (iteration). r.z listed last -> outermost.
            RDom r(0, n, 0, n, 1, (int)M, "r");
            // Ring slots (all provably in [0, 3)): current column, and the two lags.
            // Offsets are +2/+1 so the arithmetic stays non-negative.
            Expr cur = r.z % 3;       // slot for column k
            Expr c1 = (r.z + 2) % 3;  // slot for column k-1
            Expr c2 = (r.z + 1) % 3;  // slot for column k-2
            Expr km1 = r.z - 1;       // TRUE index k-1, for the coefficient tables
            Expr once = (Expr(1.0) + omega(km1)) * X(r.x, c1) - omega(km1) * X(r.x, c2) +
                        alpha(km1) * b(r.x);
            Expr matvec = alpha(km1) * A(r.x, r.y) * X(r.y, c1);
            // The ring reuses slots, so column `cur` still holds the stale x_{k-3}; we
            // cannot lean on the pure-def zero. Instead seed the mat-vec accumulation at
            // r.y == 0 with a plain assignment (r.y is the first, outer-of-r.x loop),
            // then accumulate for r.y > 0.
            X(r.x, cur) = select(r.y == 0,
                                 once - matvec,
                                 X(r.x, cur) - matvec);

            x(t) = X(t, (int)M % 3);  // endpoint extract: x_M lives in slot M % 3

            X.compute_root();
            X.update(0).allow_race_conditions().vectorize(r.x, 4);
        }

        // Output is length n.
        x.dim(0).set_bounds(0, A.dim(0).extent());
    }
};

HALIDE_REGISTER_GENERATOR(ChebyshevInductive, chebyshev_inductive)
