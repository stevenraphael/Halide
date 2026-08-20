// Standalone JIT test for merge_unsorted (bottom-up merge sort of A[x],B[x]
// leaves). Mirrors the function in merge_generator.cpp exactly.

#include "Halide.h"
#include "halide_benchmark.h"
#include <omp.h>
#include <algorithm>
#include <execution>
#include <cstdio>
#include <vector>

using namespace Halide;
using std::vector;

Func merge_unsorted(Func A, int nA, int nstages, Var k) {
    vector<Func> stages;
    int cursize = 1;
    for (int i = 0; i < nstages; i++) {
        Func stage("stage" + std::to_string(i));
        Func istage(Int(32), "istage" + std::to_string(i));
        Var j("j"), x("x");

        // Element of the left / right sub-run at local index idx. Stage 0 merges
        // the adjacent singletons A[2x], A[2x+1]; later stages merge the two
        // child runs of the previous stage.
        auto aval = [&](Expr idx) {
            return i == 0 ? A(2 * x + idx)
                          : stages[i - 1](unsafe_promise_clamped(idx, 0, cursize - 1), 2 * x);
        };
        auto bval = [&](Expr idx) {
            return i == 0 ? A(2 * x + 1 + idx)
                          : stages[i - 1](unsafe_promise_clamped(idx, 0, cursize - 1), 2 * x + 1);
        };
        // Take from the left run at (a) unless it is exhausted or the right head
        // is smaller (right not exhausted).
        auto cmp = [&](Expr a, Expr b) {
            return (a < cursize) && ((b >= cursize) || (aval(a) <= bval(b)));
        };

        Expr ip = istage(j - 1, x), jp = (j - 1) - ip;
        istage(j, x) = select(j <= 0, 0, likely(ip + select(cmp(ip, jp), 1, 0)));
        Expr ik = istage(j, x), jk = j - istage(j, x);
        stage(j, x) = select(cmp(ik, jk), aval(ik), bval(jk));

        int runlen = 2 * cursize;
        int nblocks = std::max(1, nA / runlen);
        int grain = std::max(1, nblocks / 128);
        Var xo("xo"), xi("xi");
        stage.compute_root().split(x, xo, xi, grain);//.parallel(xo);
        // Fuse istage into stage's j loop (compute per-j), recurrence buffer
        // stored per parallel task.
        istage.compute_at(stage, j).store_at(stage, xi).fold_storage(j, 1);
        cursize *= 2;
        stages.push_back(stage);
    }
    Func output("output");
    output(k) = stages.back()(k, 0);
    return output;
}

// Shared-comparison variant: istage is an inductive Tuple {count, decision}.
// The merge decision at each position is computed ONCE (component 1) and reused
// by BOTH istage's own count step (via decision(j-1)) and stage (via
// decision(j)) -- so the same comparison serves both, unlike the plain version
// which recomputes it as tookA and takeA. istage stays a separate func from the
// value gather (stage).
Func merge_unsorted_sharedcmp(Func A, int nA, int nstages, Var k) {
    vector<Func> stages;
    int cursize = 1;
    for (int i = 0; i < nstages; i++) {
        Func stage("cstage" + std::to_string(i));
        Func istage({Int(32), Int(32)}, "cistage" + std::to_string(i));  // {count, decision}
        Var j("j"), x("x");
        auto aval = [&](Expr idx) {
            return i == 0 ? A(2 * x + idx)
                          : stages[i - 1](unsafe_promise_clamped(idx, 0, cursize - 1), 2 * x);
        };
        auto bval = [&](Expr idx) {
            return i == 0 ? A(2 * x + 1 + idx)
                          : stages[i - 1](unsafe_promise_clamped(idx, 0, cursize - 1), 2 * x + 1);
        };

        Expr ip = istage(j - 1, x)[0];   // count before position j
        Expr dp = istage(j - 1, x)[1];   // decision at position j-1 (0/1)
        Expr count = select(j <= 0, 0, likely(ip + dp));  // count reuses decision(j-1)
        Expr a = ip + dp, b = j - (ip + dp);              // pointers at position j
        Expr dec = select(j <= 0,
                          select(aval(0) <= bval(0), 1, 0),
                          likely(select((a < cursize) && ((b >= cursize) || (aval(a) <= bval(b))), 1, 0)));
        istage(j, x) = Tuple(count, dec);

        // stage reuses the SAME decision stored by istage; no recomputed compare.
        Expr ik = istage(j, x)[0], dk = istage(j, x)[1];
        stage(j, x) = select(dk != 0, aval(ik), bval(j - ik));

        int runlen = 2 * cursize;
        int nblocks = std::max(1, nA / runlen);
        int grain = std::max(1, nblocks / 128);
        Var xo("xo"), xi("xi");
        stage.compute_root().split(x, xo, xi, grain).parallel(xo);
        istage.compute_at(stage, xi).store_at(stage, xo);
        cursize *= 2;
        stages.push_back(stage);
    }
    Func output("coutput");
    output(k) = stages.back()(k, 0);
    return output;
}

// Same algorithm, but istage is an ordinary serial RDom scan computed entirely
// (compute_root) before stage, i.e. NO inductive func. stage then reads istage
// as a plain load.
Func merge_unsorted_scan(Func A, int nA, int nstages, Var k) {
    vector<Func> stages;
    int cursize = 1;
    for (int i = 0; i < nstages; i++) {
        int runlen = 2 * cursize;
        Func stage(Int(32), "sstage" + std::to_string(i));
        Func istage(Int(32), "sistage" + std::to_string(i));
        Var j("j"), x("x");
        istage(j, x) = 0;  // base: 0 elements taken before position 0
        RDom r(1, runlen - 1, "r");
        Expr ip = istage(r - 1, x);
        Expr jp = (r - 1) - ip;
        Expr tookA;
        if (i == 0) {
            tookA = (ip < cursize) && ((jp >= cursize) || (A(2 * x + ip) <= A(2 * x + 1 + jp)));
        } else {
            Expr ipc = unsafe_promise_clamped(ip, 0, cursize - 1), jpc = unsafe_promise_clamped(jp, 0, cursize - 1);
            tookA = (ip < cursize) && ((jp >= cursize) || (stages[i - 1](ipc, 2 * x) <= stages[i - 1](jpc, 2 * x + 1)));
        }
        istage(r, x) = ip + select(tookA, 1, 0);

        Expr ik = istage(j, x), jk = j - istage(j, x);
        Expr takeA;
        if (i == 0) {
            takeA = (ik < cursize) && ((jk >= cursize) || (A(2 * x + ik) <= A(2 * x + 1 + jk)));
            stage(j, x) = select(takeA, A(2 * x + ik), A(2 * x + 1 + jk));
        } else {
            Expr ikc = unsafe_promise_clamped(ik, 0, cursize - 1), jkc = unsafe_promise_clamped(jk, 0, cursize - 1);
            takeA = (ik < cursize) && ((jk >= cursize) || (stages[i - 1](ikc, 2 * x) <= stages[i - 1](jkc, 2 * x + 1)));
            stage(j, x) = select(takeA, stages[i - 1](ikc, 2 * x), stages[i - 1](jkc, 2 * x + 1));
        }
        int nblocks = std::max(1, nA / runlen);
        int grain = std::max(1, nblocks / 128);
        Var xo("xo"), xi("xi");
        istage.compute_root().split(x, xo, xi, grain).parallel(xo);  // computed before stage
        istage.update(0).split(x, xo, xi, grain).parallel(xo);       // r serial; xo parallel
        stage.compute_root().split(x, xo, xi, grain).parallel(xo);
        cursize *= 2;
        stages.push_back(stage);
    }
    Func output("noind_output");
    output(k) = stages.back()(k, 0);
    return output;
}

// Tuple-valued variant (NO inductive funcs): each stage is ONE Tuple func
// s(j,x) = {count, value}, with the running count computed by an ordinary serial
// RDom update (like merge_unsorted_scan) and the merged value produced in the
// same update from that count. This fuses the two funcs of the scan version into
// a single func/buffer -- count and value stored interleaved, one pass.
Func merge_unsorted_tuple(Func A, int nA, int nstages, Var k) {
    vector<Func> stages;
    int cursize = 1;
    for (int i = 0; i < nstages; i++) {
        int runlen = 2 * cursize;
        Func s({Int(32), Int(32)}, "tstage" + std::to_string(i));
        Var j("j"), x("x");

        // value at count==0 (position 0): pick the smaller of the two run heads.
        Expr vbase;
        if (i == 0) {
            vbase = select(A(2 * x) <= A(2 * x + 1), A(2 * x), A(2 * x + 1));
        } else {
            Expr l0 = stages[i - 1](0, 2 * x)[1], r0 = stages[i - 1](0, 2 * x + 1)[1];
            vbase = select(l0 <= r0, l0, r0);
        }
        s(j, x) = Tuple(0, vbase);  // base: count 0, value at position 0

        RDom r(1, runlen - 1, "r");
        Expr ip = s(r - 1, x)[0];
        Expr jp = (r - 1) - ip;
        Expr count, value;
        if (i == 0) {
            Expr tookA = (ip < cursize) && ((jp >= cursize) || (A(2 * x + ip) <= A(2 * x + 1 + jp)));
            count = ip + select(tookA, 1, 0);
            Expr ik = count, jk = r - count;
            Expr takeA = (ik < cursize) && ((jk >= cursize) || (A(2 * x + ik) <= A(2 * x + 1 + jk)));
            value = select(takeA, A(2 * x + ik), A(2 * x + 1 + jk));
        } else {
            Expr ipc = unsafe_promise_clamped(ip, 0, cursize - 1), jpc = unsafe_promise_clamped(jp, 0, cursize - 1);
            Expr tookA = (ip < cursize) && ((jp >= cursize) || (stages[i - 1](ipc, 2 * x)[1] <= stages[i - 1](jpc, 2 * x + 1)[1]));
            count = ip + select(tookA, 1, 0);
            Expr ik = count, jk = r - count;
            Expr ikc = unsafe_promise_clamped(ik, 0, cursize - 1), jkc = unsafe_promise_clamped(jk, 0, cursize - 1);
            Expr takeA = (ik < cursize) && ((jk >= cursize) || (stages[i - 1](ikc, 2 * x)[1] <= stages[i - 1](jkc, 2 * x + 1)[1]));
            value = select(takeA, stages[i - 1](ikc, 2 * x)[1], stages[i - 1](jkc, 2 * x + 1)[1]);
        }
        s(r, x) = Tuple(count, value);

        int nblocks = std::max(1, nA / runlen);
        int grain = std::max(1, nblocks / 128);
        Var xo("xo"), xi("xi");
        s.compute_root().split(x, xo, xi, grain).parallel(xo);        // pure init
        s.update(0).split(x, xo, xi, grain).parallel(xo);             // r serial; x parallel
        cursize *= 2;
        stages.push_back(s);
    }
    Func output("tout");
    output(k) = stages.back()(k, 0)[1];
    return output;
}

// Fused inductive variant: ONE inductive Tuple func per stage, T(j,x) = {count,
// value}, using a SINGLE comparison per step for both the running count and the
// emitted value (vs the 2 comparisons the split istage/stage schedules do). The
// comparison that advances position j-1 -> j also decides value(j-1), so value
// is stored shifted by one: T(j)[1] == value at position j-1, and reads/output
// use a +1 offset.
Func merge_unsorted_fused(Func A, int nA, int nstages, Var k) {
    vector<Func> stages;
    int cursize = 1;
    for (int i = 0; i < nstages; i++) {
        Func T({Int(32), Int(32)}, "fstage" + std::to_string(i));
        Var j("j"), x("x");
        Expr ip = T(j - 1, x)[0];       // count reached before producing pos j-1
        Expr jp = (j - 1) - ip;
        Expr Aval, Bval, cmp;
        if (i == 0) {
            Aval = A(2 * x + ip);
            Bval = A(2 * x + 1 + jp);
        } else {
            Expr ipc = unsafe_promise_clamped(ip, 0, cursize - 1);
            Expr jpc = unsafe_promise_clamped(jp, 0, cursize - 1);
            Aval = stages[i - 1](ipc + 1, 2 * x)[1];       // +1: shifted value store
            Bval = stages[i - 1](jpc + 1, 2 * x + 1)[1];
        }
        cmp = (ip < cursize) && ((jp >= cursize) || (Aval <= Bval));  // the ONE comparison
        Expr count = select(j <= 0, 0, likely(ip + select(cmp, 1, 0)));
        // Value via arithmetic mask (not select(cmp,Aval,Bval)): keeps the
        // recursive ref out of select branches, which the inductive analyzer
        // forbids. Still the SAME single comparison cmp.
        Expr value = select(j <= 0, 0, likely(Bval + select(cmp, 1, 0) * (Aval - Bval)));
        T(j, x) = Tuple(count, value);

        int runlen = 2 * cursize;
        int nblocks = std::max(1, nA / runlen);
        int grain = std::max(1, nblocks / 128);
        Var xo("xo"), xi("xi");
        T.compute_root().split(x, xo, xi, grain).parallel(xo);
        cursize *= 2;
        stages.push_back(T);
    }
    Func output("fout");
    output(k) = stages.back()(k + 1, 0)[1];  // +1: undo the value shift
    return output;
}

// Fused RDom variant: same single-comparison fusion as merge_unsorted_fused,
// but the running count is an ordinary serial RDom update (NOT an inductive
// func). One Tuple func s(j,x) = {count, value}; each update step does ONE
// comparison used for both the count increment and the value emitted at that
// step (stored shifted by one, like the inductive fused version). No inductive-
// analyzer restrictions, so value can use a plain select.
Func merge_unsorted_fused_rdom(Func A, int nA, int nstages, Var k) {
    vector<Func> stages;
    int cursize = 1;
    for (int i = 0; i < nstages; i++) {
        int runlen = 2 * cursize;
        Func s({Int(32), Int(32)}, "rfstage" + std::to_string(i));
        Var j("j"), x("x");
        s(j, x) = Tuple(undef<int>(), undef<int>());  // undef -> no full init pass
        s(0, x) = Tuple(0, 0);                        // base count only (j=0 slice)

        RDom r(1, runlen, "r");  // produce count(r) and value(r-1)
        Expr ip = s(r - 1, x)[0];
        Expr jp = (r - 1) - ip;
        Expr Aval, Bval;
        if (i == 0) {
            Aval = A(2 * x + ip);
            Bval = A(2 * x + 1 + jp);
        } else {
            Expr ipc = unsafe_promise_clamped(ip, 0, cursize - 1);
            Expr jpc = unsafe_promise_clamped(jp, 0, cursize - 1);
            Aval = stages[i - 1](ipc + 1, 2 * x)[1];       // +1 shifted value store
            Bval = stages[i - 1](jpc + 1, 2 * x + 1)[1];
        }
        Expr cmp = (ip < cursize) && ((jp >= cursize) || (Aval <= Bval));  // ONE comparison
        s(r, x) = Tuple(ip + select(cmp, 1, 0), select(cmp, Aval, Bval));

        int nblocks = std::max(1, nA / runlen);
        int grain = std::max(1, nblocks / 128);
        Var xo("xo"), xi("xi");
        s.compute_root();
        s.update(0).split(x, xo, xi, grain).parallel(xo);  // base slice
        s.update(1).split(x, xo, xi, grain).parallel(xo);  // r serial; x parallel
        cursize *= 2;
        stages.push_back(s);
    }
    Func output("rfout");
    output(k) = stages.back()(k + 1, 0)[1];  // +1: undo the value shift
    return output;
}

struct Reporter : public Halide::CompileTimeErrorReporter {
    void warning(const char *msg) override { fprintf(stderr, "WARN: %s\n", msg); }
    void error(const char *msg) override {
        fprintf(stderr, "COMPILE ERROR: %s\n", msg);
        exit(1);
    }
};

// Plain single-threaded bottom-up merge sort (the same algorithm the Halide
// pipeline implements), sorting src into a fresh buffer using one scratch.
void bottom_up_mergesort(const std::vector<int> &src, std::vector<int> &buf) {
    int n = (int)src.size();
    buf = src;
    std::vector<int> tmp(n);
    for (int width = 1; width < n; width *= 2) {
        for (int lo = 0; lo < n; lo += 2 * width) {
            int mid = std::min(lo + width, n);
            int hi = std::min(lo + 2 * width, n);
            int i = lo, j = mid, k = lo;
            while (i < mid && j < hi) {  // branchless step: cmov + conditional increment
                int va = buf[i], vb = buf[j];
                bool ta = va <= vb;
                tmp[k++] = ta ? va : vb;
                i += ta;
                j += !ta;
            }
            while (i < mid) tmp[k++] = buf[i++];
            while (j < hi) tmp[k++] = buf[j++];
        }
        std::swap(buf, tmp);
    }
}

// Parallel bottom-up merge sort that mirrors the Halide grain schedule exactly:
// the same log2(n) width passes, and within each pass the independent block
// merges are grouped into ~128 coarse tasks (grain = nblocks/128 blocks per
// task) instead of one task per block -- the direct analogue of
// .split(x, xo, xi, grain).parallel(xo).
void parallel_bottom_up_mergesort(const std::vector<int> &src, std::vector<int> &buf) {
    int n = (int)src.size();
    buf = src;
    std::vector<int> tmp(n);
    int *a = buf.data(), *t = tmp.data();
    // Cap threads to the fast cores: on this heterogeneous Core Ultra 9 185H,
    // barrier-synced OpenMP over all 22 logical threads stalls on the slow
    // LP-E cores. 16 is the sweet spot.
    int nt = std::min(16, omp_get_max_threads());
    // One parallel region for the whole sort (warm thread pool, like Halide's
    // runtime) instead of re-forking a region per width pass.
#pragma omp parallel num_threads(nt)
    {
        for (int width = 1; width < n; width *= 2) {
            int step = 2 * width;
            int nblocks = (n + step - 1) / step;
            int grain = std::max(1, nblocks / 128);
            int ntasks = (nblocks + grain - 1) / grain;
#pragma omp for schedule(static)  // implicit barrier at the end
            for (int g = 0; g < ntasks; g++) {
                int b0 = g * grain, b1 = std::min(nblocks, b0 + grain);
                for (int b = b0; b < b1; b++) {  // this task's chunk of blocks, serial
                    int lo = b * step;
                    int mid = std::min(lo + width, n);
                    int hi = std::min(lo + step, n);
                    int i = lo, j = mid, k = lo;
                    while (i < mid && j < hi) {  // branchless step
                        int va = a[i], vb = a[j];
                        bool ta = va <= vb;
                        t[k++] = ta ? va : vb;
                        i += ta;
                        j += !ta;
                    }
                    while (i < mid) t[k++] = a[i++];
                    while (j < hi) t[k++] = a[j++];
                }
            }
#pragma omp single  // swap buffers once; implicit barrier makes it visible to all
            { std::swap(a, t); }
        }
    }
    if (a != buf.data()) buf = tmp;  // land result in buf
}

// Same parallel bottom-up structure, but does the merge the Halide/rank-based
// way: materialize a full-length count array via a serial scan per block (with
// the per-element edge guard), then gather values from it. This is the C
// analogue of istage/stage -- count-in-an-array + 2 comparisons/element. Used to
// isolate whether the count array (memory) is what costs Halide.
void parallel_mergesort_countarray(const std::vector<int> &src, std::vector<int> &buf) {
    int n = (int)src.size();
    buf = src;
    std::vector<int> tmp(n);
    int *a = buf.data(), *t = tmp.data();
    int nt = std::min(16, omp_get_max_threads());
#pragma omp parallel num_threads(nt)
    {
        for (int width = 1; width < n; width *= 2) {
            int step = 2 * width;
            int nblocks = (n + step - 1) / step;
            int grain = std::max(1, nblocks / 128);
            int ntasks = (nblocks + grain - 1) / grain;
#pragma omp for schedule(static)
            for (int g = 0; g < ntasks; g++) {
                int b0 = g * grain, b1 = std::min(nblocks, b0 + grain);
                for (int b = b0; b < b1; b++) {
                    int lo = b * step;
                    int mid = std::min(lo + width, n);
                    int hi = std::min(lo + step, n);
                    int la = mid - lo, lb = hi - mid, rl = hi - lo;
                    // size-2 rolling count buffer; scan and gather fused.
                    int c[2];
                    c[0] = 0;  // count at position 0
                    for (int p = 0; p < rl; p++) {
                        if (p > 0) {  // advance count from previous position
                            int ap = c[(p - 1) & 1], bp = (p - 1) - ap;
                            bool took = (ap < la) && ((bp >= lb) || (a[lo + ap] <= a[mid + bp]));
                            c[p & 1] = ap + (took ? 1 : 0);
                        }
                        int ak = c[p & 1], bk = p - ak;
                        bool take = (ak < la) && ((bk >= lb) || (a[lo + ak] <= a[mid + bk]));
                        t[lo + p] = take ? a[lo + ak] : a[mid + bk];
                    }
                }
            }
#pragma omp single
            { std::swap(a, t); }
        }
    }
    if (a != buf.data()) buf = tmp;
}

int main() {
    static Reporter reporter;
    Halide::set_custom_compile_time_error_reporter(&reporter);
    // Same fix as the C version: cap Halide's thread pool to the fast cores
    // instead of all 22 logical (heterogeneous Core Ultra 9 185H).
    setenv("HL_NUM_THREADS", "16", 1);
    const int n = 4 * 1024 * 1024;  // single unsorted array, length == 2^nstages
    const int nstages = 22;         // 2^nstages == n
    std::vector<int> a(n);
    srand(12345);
    for (int t = 0; t < n; t++) { a[t] = rand() % 100000; }

    Buffer<int> A(n);
    for (int t = 0; t < n; t++) { A(t) = a[t]; }

    Var k("k");
    Func Af("Af");
    // Clamp speculative boundary reads.
    Af(k) = A(clamp(k, 0, n - 1));

    Func out = merge_unsorted(Af, n, nstages, k);
    out.compile_jit();  // exclude JIT compilation from the timing

    Func out_scan = merge_unsorted_scan(Af, n, nstages, k);
    out_scan.compile_jit();

    Func out_tuple = merge_unsorted_tuple(Af, n, nstages, k);
    out_tuple.compile_jit();

    Func out_fused = merge_unsorted_fused(Af, n, nstages, k);
    out_fused.compile_jit();

    Func out_frdom = merge_unsorted_fused_rdom(Af, n, nstages, k);
    out_frdom.compile_jit();

    Func out_shared = merge_unsorted_sharedcmp(Af, n, nstages, k);
    out_shared.compile_jit();

    Buffer<int> result(n), result_scan(n), result_tuple(n), result_fused(n), result_frdom(n), result_shared(n);
    out.realize(result);
    out_scan.realize(result_scan);
    out_tuple.realize(result_tuple);
    out_fused.realize(result_fused);
    out_frdom.realize(result_frdom);
    out_shared.realize(result_shared);

    std::vector<int> reference = a;
    std::sort(reference.begin(), reference.end());

    // Benchmark: Halide merge_unsorted vs std::sort of the same array.
    double halide_t = Halide::Tools::benchmark(3, 10, [&]() { out.realize(result); });
    double scan_t = Halide::Tools::benchmark(3, 10, [&]() { out_scan.realize(result_scan); });
    double tuple_t = Halide::Tools::benchmark(3, 10, [&]() { out_tuple.realize(result_tuple); });
    double fused_t = Halide::Tools::benchmark(3, 10, [&]() { out_fused.realize(result_fused); });
    double frdom_t = Halide::Tools::benchmark(3, 10, [&]() { out_frdom.realize(result_frdom); });
    double shared_t = Halide::Tools::benchmark(3, 10, [&]() { out_shared.realize(result_shared); });
    std::vector<int> scratch(n), buf;
    double std_sort_t = Halide::Tools::benchmark(3, 10, [&]() {
        std::copy(a.begin(), a.end(), scratch.begin());
        std::sort(scratch.begin(), scratch.end());
    });
    double std_stable_t = Halide::Tools::benchmark(3, 10, [&]() {
        std::copy(a.begin(), a.end(), scratch.begin());
        std::stable_sort(scratch.begin(), scratch.end());  // merge sort
    });
    double mergesort_t = Halide::Tools::benchmark(3, 10, [&]() {
        bottom_up_mergesort(a, buf);
    });
    std::vector<int> pbuf;
    double pmergesort_t = Halide::Tools::benchmark(3, 10, [&]() {
        parallel_bottom_up_mergesort(a, pbuf);
    });
    std::vector<int> cabuf;
    double camergesort_t = Halide::Tools::benchmark(3, 10, [&]() {
        parallel_mergesort_countarray(a, cabuf);
    });
    for (int t = 0; t < n; t++) {
        if (cabuf[t] != reference[t]) { printf("MISMATCH (countarray) at %d\n", t); return -1; }
    }
    double par_sort_t = Halide::Tools::benchmark(3, 10, [&]() {
        std::copy(a.begin(), a.end(), scratch.begin());
        std::sort(std::execution::par_unseq, scratch.begin(), scratch.end());
    });
    double par_stable_t = Halide::Tools::benchmark(3, 10, [&]() {
        std::copy(a.begin(), a.end(), scratch.begin());
        std::stable_sort(std::execution::par, scratch.begin(), scratch.end());
    });
    // verify parallel C merge sort
    for (int t = 0; t < n; t++) {
        if (pbuf[t] != reference[t]) { printf("MISMATCH (omp) at %d\n", t); return -1; }
    }
    printf("elements: %d\n", n);
    printf("Halide merge_unsorted (induct):%.3f ms\n", halide_t * 1e3);
    printf("Halide merge_unsorted (scan):  %.3f ms\n", scan_t * 1e3);
    printf("Halide merge_unsorted (tuple): %.3f ms\n", tuple_t * 1e3);
    printf("Halide merge_unsorted (fused): %.3f ms\n", fused_t * 1e3);
    printf("Halide merge_unsorted (frdom): %.3f ms\n", frdom_t * 1e3);
    printf("Halide merge_unsorted (shared):%.3f ms\n", shared_t * 1e3);
    printf("C mergesort (OpenMP, %d thr):   %.3f ms\n", std::min(16, omp_get_max_threads()), pmergesort_t * 1e3);
    printf("C mergesort COUNT-ARRAY (%d thr):%.3f ms\n", std::min(16, omp_get_max_threads()), camergesort_t * 1e3);
    printf("std::sort par_unseq (TBB):     %.3f ms\n", par_sort_t * 1e3);
    printf("std::stable_sort par (TBB):    %.3f ms\n", par_stable_t * 1e3);
    printf("C bottom-up mergesort (1 thr): %.3f ms\n", mergesort_t * 1e3);
    printf("std::stable_sort (merge, 1thr):%.3f ms\n", std_stable_t * 1e3);
    printf("std::sort (introsort, 1 thr):  %.3f ms\n", std_sort_t * 1e3);

    /*printf("first 16: ");
    for (int t = 0; t < 16; t++) printf("%d ", result(t));
    printf("\nlast 8:   ");
    for (int t = n - 8; t < n; t++) printf("%d ", result(t));
    printf("\n");

    for (int t = 0; t < n; t++) {
        if (result(t) != reference[t]) {
            printf("MISMATCH (induct) at %d: got %d want %d\n", t, result(t), reference[t]);
            return -1;
        }
        if (result_scan(t) != reference[t]) {
            printf("MISMATCH (scan) at %d: got %d want %d\n", t, result_scan(t), reference[t]);
            return -1;
        }
        if (result_tuple(t) != reference[t]) {
            printf("MISMATCH (tuple) at %d: got %d want %d\n", t, result_tuple(t), reference[t]);
            return -1;
        }
        if (result_fused(t) != reference[t]) {
            printf("MISMATCH (fused) at %d: got %d want %d\n", t, result_fused(t), reference[t]);
            return -1;
        }
        if (result_frdom(t) != reference[t]) {
            printf("MISMATCH (frdom) at %d: got %d want %d\n", t, result_frdom(t), reference[t]);
            return -1;
        }
        if (result_shared(t) != reference[t]) {
            printf("MISMATCH (shared) at %d: got %d want %d\n", t, result_shared(t), reference[t]);
            return -1;
        }
    }*/
    printf("Success!\n");
    return 0;
}
