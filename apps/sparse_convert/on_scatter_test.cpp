// Verify the O(N) combined-func RDom cursor scatter for CSR->CSC.
//
// Key idea: put the running per-column cursor AND both output arrays into ONE
// func `f`, laid out along a single dimension:
//     f[0 .. C)          = cursor (one write position per column)
//     f[C .. C+N)        = csc_row output
//     f[C+N .. C+2N)     = csc_val output
// Then a SINGLE update definition over a 2-D RDom (k = nonzero, m = sub-step)
// does, per nonzero, in order:
//     m=0: d = f[col];  f[C   + d] = row
//     m=1: d = f[col];  f[C+N + d] = val
//     m=2:              f[col]     = f[col] + 1   (bump)
// Within one update def, reads of f see earlier iterations' writes, so the bump
// at m=2 is visible to the next nonzero. O(1) per nonzero -> O(N) total, O(C)
// extra state. No per-column loop, so it does NOT scale with C.

#include "Halide.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    int R = argc > 1 ? atoi(argv[1]) : 65536;
    int C = argc > 2 ? atoi(argv[2]) : 512;
    double density = argc > 3 ? atof(argv[3]) : 0.005;
    int N = (int)(density * (double)R * (double)C);

    Buffer<int> coo_row(N), coo_col(N), coo_val(N), row_add(R);
    srand(12345);
    std::vector<int> rows(N);
    for (int k = 0; k < N; k++) rows[k] = rand() % R;
    std::sort(rows.begin(), rows.end());
    for (int k = 0; k < N; k++) {
        coo_row(k) = rows[k];
        coo_col(k) = rand() % C;
        coo_val(k) = rand() % 1000;
    }
    for (int r = 0; r < R; r++) row_add(r) = r;

    try {
        Var i("i"), k("k"), c("c");
        RDom kn(0, N);

        // CSR arrays (row-sorted COO) + per-row add.
        Func csr_col("csr_col"), csr_val("csr_val");
        csr_col(k) = coo_col(k);
        csr_val(k) = coo_val(k) + row_add(clamp(coo_row(k), 0, R - 1));
        csr_col.compute_root();
        csr_val.compute_root();

        // col_count + col_ptr (exclusive prefix sum), materialized (tiny).
        Func col_count("col_count");
        col_count(c) = 0;
        col_count(clamp(csr_col(kn), 0, C - 1)) += 1;
        col_count.compute_root();
        Func col_ptr(Int(32), "col_ptr");
        col_ptr(i) = undef<int>();
        col_ptr(0) = 0;
        RDom rc(1, C);
        col_ptr(rc) = col_ptr(rc - 1) + col_count(clamp(rc - 1, 0, C - 1));
        col_ptr.compute_root();

        // Combined func: cursor | row out | val out.
        Func f(Int(32), "f");
        f(i) = select(i < C, col_ptr(clamp(i, 0, C)), 0);

        // Last-listed RDom dim is outermost, so put k last (outer) and m
        // first (inner): iterate all 3 sub-steps of one nonzero before moving on.
        RDom r(0, 3, 0, N);          // r.x = sub-step m, r.y = nonzero k
        Expr kk = r.y, m = r.x;
        Expr col = clamp(csr_col(kk), 0, C - 1);
        Expr d = clamp(f(col), 0, N - 1);              // current cursor for col
        Expr idx = select(m == 0, C + d,
                          m == 1, C + N + d,
                          col);                         // m==2 -> cursor slot
        Expr val = select(m == 0, coo_row(kk),
                          m == 1, csr_val(kk),
                          f(col) + 1);                  // m==2 -> bump
        f(clamp(idx, 0, C + 2 * N - 1)) = val;
        f.compute_root();

        // Pull the outputs out through wrapper funcs.
        Func row_out("row_out"), val_out("val_out");
        row_out(k) = f(C + k);
        val_out(k) = f(C + N + k);

        // Time it.
        row_out.compile_jit();
        val_out.compile_jit();
        Buffer<int> ro(N), vo(N);
        row_out.realize(ro);
        val_out.realize(vo);
        const int trials = 20;
        double best = 1e18;
        for (int t = 0; t < trials; t++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            row_out.realize(ro);
            val_out.realize(vo);
            auto t1 = std::chrono::high_resolution_clock::now();
            best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        // Ground truth + timed C++ cursor baseline (col_ptr prefix sum + scatter).
        std::vector<int> cp(C + 1, 0), cur(C + 1), gr(N), gv(N), pre(N);
        for (int kk2 = 0; kk2 < N; kk2++) pre[kk2] = coo_val(kk2) + row_add(coo_row(kk2));
        double cbest = 1e18;
        for (int t = 0; t < trials; t++) {
            std::fill(cp.begin(), cp.end(), 0);
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int kk2 = 0; kk2 < N; kk2++) cp[coo_col(kk2) + 1]++;
            for (int cc = 0; cc < C; cc++) cp[cc + 1] += cp[cc];
            for (int cc = 0; cc <= C; cc++) cur[cc] = cp[cc];
            for (int kk2 = 0; kk2 < N; kk2++) {
                int col2 = coo_col(kk2);
                int d2 = cur[col2]++;
                gr[d2] = coo_row(kk2);
                gv[d2] = pre[kk2];
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            cbest = std::min(cbest, std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        bool ok = true;
        for (int kk2 = 0; kk2 < N && ok; kk2++)
            if (ro(kk2) != gr[kk2] || vo(kk2) != gv[kk2]) ok = false;

        printf("R=%d C=%d N=%d (N/R*C=%.4f)\n", R, C, N, (double)N / ((double)R * C));
        printf("  Halide O(N) combined-RDom scatter: %8.3f ms\n", best);
        printf("  C++ cursor (full: hist+psum+scatter): %8.3f ms\n", cbest);
        printf("  agree: %s\n", ok ? "yes" : "NO");
        return ok ? 0 : 1;
    } catch (const Halide::Error &e) {
        printf("HALIDE ERROR: %s\n", e.what());
        return 2;
    }
}
