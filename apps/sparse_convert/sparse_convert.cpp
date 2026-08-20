// Sparse-matrix format conversion pipeline expressed in Halide, mixing
// inductive functions (for the prefix-sum "spines") with ordinary RDom
// funcs (for the data-dependent scatters).
//
// Pipeline:
//   1. COO (row-sorted) -> CSR      (histogram + prefix sum)
//   2. per-row scalar add           (add row_add[r] to every nonzero in row r)
//   3. CSR -> CSC                   (histogram + prefix sum + cursor scatter)
//
// The two prefix sums (row_ptr, col_ptr) are inductive functions, each fused
// into its consumer. Everything else is a pure/RDom func. The point of this
// prototype is to show exactly where the inductive/RDom boundary lands.
//
// Build (JIT), from the Halide tree root:
//   g++ apps/sparse_convert/sparse_convert.cpp -g \
//       -Iinclude -Lbuild/src -lHalide -lpthread -ldl -o /tmp/sparse_convert -std=c++17
//   LD_LIBRARY_PATH=build/src /tmp/sparse_convert

#include "Halide.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Halide;

int run();
int main() {
    try { return run(); }
    catch (const Halide::Error &e) { printf("HALIDE ERROR: %s\n", e.what()); return 2; }
}
int run() {
    // ------------------------------------------------------------------
    // Input COO matrix, sorted by row. Hard-coded small example so we can
    // check the result by hand.
    //   matrix (rows=4, cols=5):
    //     (0,1)=10  (0,3)=20
    //     (1,0)=30
    //     (2,2)=40  (2,3)=50  (2,4)=60
    //     (3,1)=70
    // ------------------------------------------------------------------
    const int R = 4;    // rows
    const int C = 5;    // cols
    const int N = 7;    // nonzeros

    Buffer<int> coo_row(N), coo_col(N), coo_val(N);
    {
        int rr[N] = {0, 0, 1, 2, 2, 2, 3};
        int cc[N] = {1, 3, 0, 2, 3, 4, 1};
        int vv[N] = {10, 20, 30, 40, 50, 60, 70};
        for (int k = 0; k < N; k++) {
            coo_row(k) = rr[k];
            coo_col(k) = cc[k];
            coo_val(k) = vv[k];
        }
    }

    // Per-row value to add to every nonzero in that row.
    Buffer<int> row_add(R);
    for (int r = 0; r < R; r++) row_add(r) = r * 100;  // 0,100,200,300

    Var i("i"), k("k"), r("r"), c("c");

    // ==================================================================
    // Stage 1: COO -> CSR
    // ==================================================================

    // 1a. Histogram: count nonzeros per row.  Data-dependent scatter-add,
    //     so this is an RDom update, not inductive.
    Func row_count("row_count");
    row_count(r) = 0;
    RDom kn(0, N);
    row_count(clamp(coo_row(kn), 0, R - 1)) += 1;

    // 1b. row_ptr = exclusive prefix sum of row_count, length R+1.
    //     This is the inductive func. row_ptr(0)=0; row_ptr(i)=row_ptr(i-1)+count(i-1).
    Func row_ptr(Int(32), "row_ptr");
    row_ptr(i) = select(i <= 0, 0,
                        likely(row_ptr(i - 1) + row_count(clamp(i - 1, 0, R - 1))));

    // ==================================================================
    // Stage 2: per-row scalar add on the CSR values.
    // For a row-sorted COO, the CSR value/col arrays are just the COO arrays
    // in order, so csr_val[k] = coo_val[k] + row_add[coo_row[k]].
    // ==================================================================
    Func csr_col("csr_col"), csr_val("csr_val");
    csr_col(k) = coo_col(k);
    csr_val(k) = coo_val(k) + row_add(clamp(coo_row(k), 0, R - 1));

    // ==================================================================
    // Stage 3: CSR -> CSC (transpose)
    // ==================================================================

    // 3a. Histogram: count nonzeros per column. RDom scatter-add.
    Func col_count("col_count");
    col_count(c) = 0;
    col_count(clamp(csr_col(kn), 0, C - 1)) += 1;

    // 3b. col_ptr = exclusive prefix sum of col_count, length C+1. Inductive.
    Func col_ptr(Int(32), "col_ptr");
    col_ptr(i) = select(i <= 0, 0,
                        likely(col_ptr(i - 1) + col_count(clamp(i - 1, 0, C - 1))));

    // 3c. Within-column rank, done in Halide instead of a C++ cursor.
    //     pos(c, k) = number of nonzeros j in [0, k) whose column is c.
    //     This is an inductive scan in k -- it replaces the mutable
    //     per-column cursor that no single Halide update def can express
    //     (a cursor read-modify-write would have to write two different
    //     addresses in one interleaved step).
    Func pos(Int(32), "pos");
    pos(c, k) = select(k <= 0, 0,
                       likely(pos(c, k - 1) +
                              select(clamp(csr_col(clamp(k - 1, 0, N - 1)), 0, C - 1) == c, 1, 0)));

    // dest(k) = col_ptr[col(k)] + pos(col(k), k). Destinations are unique
    // (it's a permutation), so the following scatter is order-independent.
    Func dest(Int(32), "dest");
    Expr colk = clamp(csr_col(k), 0, C - 1);
    dest(k) = col_ptr(colk) + pos(colk, k);

    // For a row-sorted COO, the source row of nonzero k is just coo_row(k).
    Func src_row("src_row");
    src_row(k) = coo_row(k);

    // 3d. Pure scatter into the CSC arrays (a plain RDom update, unique dests).
    Func csc_row(Int(32), "csc_row"), csc_val(Int(32), "csc_val");
    csc_row(i) = 0;
    csc_val(i) = 0;
    Expr d = clamp(dest(kn), 0, N - 1);
    csc_row(d) = src_row(kn);
    csc_val(d) = csr_val(kn);

    // pos is 2D over (c, k); compute it root here (O(N*C) for the demo -- the
    // inductive fold benefit is defeated when a downstream scatter needs random
    // access to the whole scan, which is exactly the transpose's situation).
    pos.compute_root();
    col_ptr.compute_root();
    row_ptr.compute_root();

    // Inductive funcs can't be output buffers, so read them through trivial
    // wrapper funcs to inspect their values.
    Func row_ptr_out("row_ptr_out"), col_ptr_out("col_ptr_out");
    row_ptr_out(i) = row_ptr(i);
    col_ptr_out(i) = col_ptr(i);

    Buffer<int> row_ptr_buf = row_ptr_out.realize({R + 1});
    Buffer<int> col_ptr_buf = col_ptr_out.realize({C + 1});
    Buffer<int> csc_row_buf = csc_row.realize({N});
    Buffer<int> csc_val_buf = csc_val.realize({N});

    // ------------------------------------------------------------------
    // Report + verify.
    // ------------------------------------------------------------------
    printf("row_ptr: ");
    for (int a = 0; a <= R; a++) printf("%d ", row_ptr_buf(a));
    printf("\ncol_ptr: ");
    for (int a = 0; a <= C; a++) printf("%d ", col_ptr_buf(a));
    printf("\ncsc (col-major, row:val): ");
    for (int a = 0; a < N; a++) printf("%d:%d ", csc_row_buf(a), csc_val_buf(a));
    printf("\n");

    // Expected row_ptr for counts [2,1,3,1] -> [0,2,3,6,7].
    int exp_rp[R + 1] = {0, 2, 3, 6, 7};
    for (int a = 0; a <= R; a++) {
        if (row_ptr_buf(a) != exp_rp[a]) { printf("row_ptr mismatch\n"); return 1; }
    }
    // Column counts: col0:1 col1:2 col2:1 col3:2 col4:1 -> col_ptr [0,1,3,4,6,7].
    int exp_cp[C + 1] = {0, 1, 3, 4, 6, 7};
    for (int a = 0; a <= C; a++) {
        if (col_ptr_buf(a) != exp_cp[a]) { printf("col_ptr mismatch\n"); return 1; }
    }

    printf("Success!\n");
    return 0;
}
