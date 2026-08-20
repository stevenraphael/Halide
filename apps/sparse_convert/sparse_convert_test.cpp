// Fast test suite: CSR->CSC conversion (COO->CSR + per-row add + transpose),
// built two ways that differ ONLY in how the col_ptr prefix sum is expressed:
//   inductive : col_ptr is an inductive func (select-guarded scan).
//   rdom      : col_ptr is a classic RDom update-def scan.
// Both share the identical O(N) combined-func cursor scatter for the transpose.
//
// For each small config we check both against a C++ cursor ground truth and
// print timings. Small N + few trials so the whole suite runs in well under a
// second.
//
// Build from the Halide tree root:
//   g++ apps/sparse_convert/sparse_convert_test.cpp -g \
//       -Iinclude -Lbuild/src -lHalide -lpthread -ldl \
//       -o /tmp/sct -std=c++17
//   LD_LIBRARY_PATH=build/src /tmp/sct

#include "Halide.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace Halide;

struct Out { Func row, val; };

// Build a full COO->CSC pipeline. `inductive` toggles only the col_ptr scan.
static Out build(bool inductive, int R, int C, int N,
                 const Buffer<int> &coo_row, const Buffer<int> &coo_col,
                 const Buffer<int> &coo_val, const Buffer<int> &row_add) {
    Var i("i"), k("k"), c("c");
    RDom kn(0, N);

    Func csr_col("csr_col"), csr_val("csr_val");
    csr_col(k) = coo_col(k);
    csr_val(k) = coo_val(k) + row_add(clamp(coo_row(k), 0, R - 1));
    csr_col.compute_root();
    csr_val.compute_root();

    Func col_count("col_count");
    col_count(c) = 0;
    col_count(clamp(csr_col(kn), 0, C - 1)) += 1;
    col_count.compute_root();

    // The scan under comparison.
    Func col_ptr(Int(32), "col_ptr");
    if (inductive) {
        col_ptr(i) = select(i <= 0, 0,
                            likely(col_ptr(i - 1) + col_count(clamp(i - 1, 0, C - 1))));
    } else {
        col_ptr(i) = undef<int>();
        col_ptr(0) = 0;
        RDom rc(1, C);
        col_ptr(rc) = col_ptr(rc - 1) + col_count(clamp(rc - 1, 0, C - 1));
    }
    col_ptr.compute_root();

    // O(N) combined-func cursor scatter: f = [cursor | row out | val out].
    Func f(Int(32), "f");
    f(i) = select(i < C, col_ptr(clamp(i, 0, C)), 0);
    RDom r(0, 3, 0, N);                 // m inner (r.x), k outer (r.y)
    Expr kk = r.y, m = r.x;
    Expr col = clamp(csr_col(kk), 0, C - 1);
    Expr d = clamp(f(col), 0, N - 1);
    Expr idx = select(m == 0, C + d, m == 1, C + N + d, col);
    Expr val = select(m == 0, coo_row(kk), m == 1, csr_val(kk), f(col) + 1);
    f(clamp(idx, 0, C + 2 * N - 1)) = val;
    f.compute_root();

    Func row_out("row_out"), val_out("val_out");
    row_out(k) = f(C + k);
    val_out(k) = f(C + N + k);
    return {row_out, val_out};
}

static double time_realize(Func rf, Func vf, Buffer<int> &rb, Buffer<int> &vb, int trials) {
    rf.compile_jit();
    vf.compile_jit();
    rf.realize(rb);
    vf.realize(vb);
    double best = 1e18;
    for (int t = 0; t < trials; t++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        rf.realize(rb);
        vf.realize(vb);
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

static int run_case(int R, int C, int N) {
    Buffer<int> coo_row(N), coo_col(N), coo_val(N), row_add(R);
    srand(R * 131 + C * 17 + N);
    std::vector<int> rows(N);
    for (int k = 0; k < N; k++) rows[k] = rand() % R;
    std::sort(rows.begin(), rows.end());
    for (int k = 0; k < N; k++) {
        coo_row(k) = rows[k];
        coo_col(k) = rand() % C;
        coo_val(k) = rand() % 1000;
    }
    for (int r = 0; r < R; r++) row_add(r) = r;

    // C++ cursor ground truth.
    std::vector<int> cp(C + 1, 0), cur(C + 1), gr(N), gv(N);
    for (int k = 0; k < N; k++) cp[coo_col(k) + 1]++;
    for (int c = 0; c < C; c++) cp[c + 1] += cp[c];
    for (int c = 0; c <= C; c++) cur[c] = cp[c];
    for (int k = 0; k < N; k++) {
        int d = cur[coo_col(k)]++;
        gr[d] = coo_row(k);
        gv[d] = coo_val(k) + row_add(coo_row(k));
    }

    const int trials = 5;
    Out ind = build(true, R, C, N, coo_row, coo_col, coo_val, row_add);
    Out rdm = build(false, R, C, N, coo_row, coo_col, coo_val, row_add);
    Buffer<int> ir(N), iv(N), rr(N), rv(N);
    double t_ind = time_realize(ind.row, ind.val, ir, iv, trials);
    double t_rdm = time_realize(rdm.row, rdm.val, rr, rv, trials);

    bool ok = true;
    for (int k = 0; k < N && ok; k++)
        if (ir(k) != gr[k] || iv(k) != gv[k] || rr(k) != gr[k] || rv(k) != gv[k]) ok = false;

    printf("  R=%-6d C=%-5d N=%-7d  inductive %6.3f ms | rdom %6.3f ms | %s\n",
           R, C, N, t_ind, t_rdm, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main() {
    struct { int R, C, N; } cases[] = {
        {256, 16, 256}, {1024, 64, 2048}, {4096, 128, 8192},
        {8192, 256, 16384}, {2048, 512, 4096}, {16384, 32, 32768},
    };
    printf("COO->CSR->CSC: inductive vs rdom col_ptr scan (shared O(N) scatter)\n");
    int fails = 0;
    for (auto &c : cases) fails += run_case(c.R, c.C, c.N);
    printf(fails ? "\n%d case(s) FAILED\n" : "\nAll cases passed\n", fails);
    return fails ? 1 : 0;
}
