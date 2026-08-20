// Benchmark: CSR -> CSC transpose (with COO->CSR + per-row add up front),
// built two ways that compute the *same* result:
//
//   inductive: row_ptr / col_ptr / pos are inductive funcs (select-guarded
//              scans, no RDom).
//   rdom:      the same three quantities via classic RDom update-def scans.
//
// Everything else (the two histogram scatter-adds and the final permutation
// scatter) is identical between the two. Both are timed end-to-end to CSC.
//
// The point is to see what inductive funcs buy for this shape. Spoiler worth
// keeping in mind: the fold_storage/fusion win needs a consumer that reads the
// scan in order; here the transpose scatter indexes col_ptr[col(k)] and
// pos(col(k),k) at data-dependent positions, so the scans must be fully
// materialized either way -- limiting the inductive advantage.
//
// Build from the Halide tree root:
//   g++ apps/sparse_convert/sparse_convert_bench.cpp -g \
//       -Iinclude -Lbuild/src -lHalide -lpthread -ldl \
//       -o /tmp/sparse_convert_bench -std=c++17
//   LD_LIBRARY_PATH=build/src /tmp/sparse_convert_bench [R] [C] [N]

#include "Halide.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Halide;

// Build the pipeline. If `inductive` is true, use inductive funcs for the
// three scans; otherwise use RDom update-def scans. Returns the two CSC
// output funcs (csc_row, csc_val) plus col_ptr_out for checking.
struct Pipe {
    Func csc_row, csc_val, col_ptr_out;
};

static Pipe build(bool inductive, int R, int C, int N,
                  const Buffer<int> &coo_row, const Buffer<int> &coo_col,
                  const Buffer<int> &coo_val, const Buffer<int> &row_add) {
    Var i("i"), k("k"), r("r"), c("c");
    RDom kn(0, N);

    // Stage 1a/3a: histograms (RDom scatter-add either way).
    Func row_count("row_count"), col_count("col_count");
    row_count(r) = 0;
    row_count(clamp(coo_row(kn), 0, R - 1)) += 1;
    Func csr_col("csr_col"), csr_val("csr_val");
    csr_col(k) = coo_col(k);
    csr_val(k) = coo_val(k) + row_add(clamp(coo_row(k), 0, R - 1));
    col_count(c) = 0;
    col_count(clamp(csr_col(kn), 0, C - 1)) += 1;
    row_count.compute_root();
    csr_col.compute_root();
    csr_val.compute_root();
    col_count.compute_root();

    // Scans: col_ptr (exclusive prefix sum) and pos (within-column rank).
    Func col_ptr(Int(32), "col_ptr"), pos(Int(32), "pos");
    if (inductive) {
        col_ptr(i) = select(i <= 0, 0,
                            likely(col_ptr(i - 1) + col_count(clamp(i - 1, 0, C - 1))));
        pos(c, k) = select(k <= 0, 0,
                           likely(pos(c, k - 1) +
                                  select(clamp(csr_col(clamp(k - 1, 0, N - 1)), 0, C - 1) == c, 1, 0)));
    } else {
        col_ptr(i) = undef<int>();
        col_ptr(0) = 0;
        RDom rc(1, C);
        col_ptr(rc) = col_ptr(rc - 1) + col_count(clamp(rc - 1, 0, C - 1));

        pos(c, k) = undef<int>();
        pos(c, 0) = 0;
        RDom rk(1, N - 1);
        pos(c, rk) = pos(c, rk - 1) +
                     select(clamp(csr_col(clamp(rk - 1, 0, N - 1)), 0, C - 1) == c, 1, 0);
        // Match the inductive func's loop order: scan axis (rk) outer, c inner
        // and contiguous. Without this the default `for c: for rk` strides the
        // scan by C elements per step and tanks cache behavior.
        pos.update(1).reorder(c, rk);
    }
    col_ptr.compute_root();  // tiny (C+1); materialized either way

    // dest(k) = col_ptr[col(k)] + pos(col(k), k); pure unique-dest scatter.
    Func dest(Int(32), "dest");
    Expr colk = clamp(csr_col(k), 0, C - 1);
    dest(k) = col_ptr(colk) + pos(colk, k);
    dest.compute_root();

    if (inductive) {
        // The whole point: pos is NOT materialized. Fuse it into dest's k loop
        // and fold the scan axis to a single C-wide slice -- storage drops from
        // N*C to C. That folded slice IS the per-column cursor.
        pos.compute_at(dest, k).store_at(dest, Var::outermost()).fold_storage(k, 1);
    } else {
        // RDom scan has no in-order consumer to fuse into: it must materialize
        // the full N*C array before the (random-access) scatter can read it.
        pos.compute_root();
    }

    Func csc_row(Int(32), "csc_row"), csc_val(Int(32), "csc_val");
    csc_row(i) = 0;
    csc_val(i) = 0;
    Expr d = clamp(dest(kn), 0, N - 1);
    csc_row(d) = coo_row(kn);
    csc_val(d) = csr_val(kn);

    Func col_ptr_out("col_ptr_out");
    col_ptr_out(i) = col_ptr(i);

    return {csc_row, csc_val, col_ptr_out};
}

static double time_it(Func csc_row, Func csc_val, int N,
                      Buffer<int> &row_buf, Buffer<int> &val_buf) {
    csc_row.compile_jit();
    csc_val.compile_jit();
    csc_row.realize(row_buf);  // warm up
    csc_val.realize(val_buf);
    const int trials = 20;
    double best = 1e18;
    for (int t = 0; t < trials; t++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        csc_row.realize(row_buf);
        csc_val.realize(val_buf);
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

// Honest non-inductive baseline: the O(N)-work, O(C)-space imperative cursor
// that any real CSC transpose uses. Times the scatter only (col_ptr prefix sum
// is trivial and shared).
static double time_cursor(int R, int C, int N,
                          const Buffer<int> &coo_row, const Buffer<int> &csr_col,
                          const Buffer<int> &csr_val, const Buffer<int> &col_ptr,
                          Buffer<int> &row_buf, Buffer<int> &val_buf) {
    std::vector<int> cur(C);
    const int trials = 20;
    double best = 1e18;
    for (int t = 0; t < trials; t++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int c = 0; c < C; c++) cur[c] = col_ptr(c);
        for (int k = 0; k < N; k++) {
            int col = csr_col(k);
            int d = cur[col]++;
            row_buf(d) = coo_row(k);
            val_buf(d) = csr_val(k);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

int main(int argc, char **argv) {
    // Realistic sparse dimensions: N is a small fraction of R*C. Defaults give
    // ~0.5% density. pos is O(N*C), so C dominates the inductive cost.
    int R = argc > 1 ? atoi(argv[1]) : 65536;
    int C = argc > 2 ? atoi(argv[2]) : 512;
    double density = argc > 3 ? atof(argv[3]) : 0.005;
    int N = (int)(density * (double)R * (double)C);

    // Generate a row-sorted COO. Rows chosen sorted, cols/vals random.
    Buffer<int> coo_row(N), coo_col(N), coo_val(N), row_add(R);
    srand(12345);
    {
        std::vector<int> rows(N);
        for (int k = 0; k < N; k++) rows[k] = rand() % R;
        std::sort(rows.begin(), rows.end());
        for (int k = 0; k < N; k++) {
            coo_row(k) = rows[k];
            coo_col(k) = rand() % C;
            coo_val(k) = rand() % 1000;
        }
        for (int r = 0; r < R; r++) row_add(r) = r;
    }

    try {
        Pipe ind = build(true, R, C, N, coo_row, coo_col, coo_val, row_add);
        Pipe rdm = build(false, R, C, N, coo_row, coo_col, coo_val, row_add);

        Buffer<int> ir(N), iv(N), rr(N), rv(N), cr(N), cv(N);
        double t_ind = time_it(ind.csc_row, ind.csc_val, N, ir, iv);
        double t_rdm = time_it(rdm.csc_row, rdm.csc_val, N, rr, rv);

        // Materialize the shared inputs the C++ cursor needs.
        Func csr_col("csr_col_o"), csr_val("csr_val_o");
        Var k("k");
        csr_col(k) = coo_col(k);
        csr_val(k) = coo_val(k) + row_add(clamp(coo_row(k), 0, R - 1));
        Buffer<int> csr_col_b = csr_col.realize({N});
        Buffer<int> csr_val_b = csr_val.realize({N});
        Buffer<int> col_ptr_b = ind.col_ptr_out.realize({C + 1});
        double t_cur = time_cursor(R, C, N, coo_row, csr_col_b, csr_val_b, col_ptr_b, cr, cv);

        bool ok = true;
        for (int k = 0; k < N && ok; k++)
            if (ir(k) != rr(k) || iv(k) != rv(k) ||
                ir(k) != cr(k) || iv(k) != cv(k)) ok = false;

        printf("R=%d C=%d N=%d (density %.3f%%, N/R*C=%.4f)\n",
               R, C, N, density * 100.0, (double)N / ((double)R * C));
        printf("  C++ cursor      (O(N) work, O(C) space):     %8.3f ms\n", t_cur);
        printf("  inductive fold  (O(N*C) work, O(C) space):   %8.3f ms\n", t_ind);
        printf("  rdom materialize(O(N*C) work, O(N*C) space): %8.3f ms\n", t_rdm);
        printf("  agree: %s\n", ok ? "yes" : "NO");
        return ok ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
