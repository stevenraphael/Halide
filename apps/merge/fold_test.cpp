// Tests for the "in-place recurrence" storage-folding optimization.
//
// Idea: a serial self-recurrence f(x) = g(..., f(x-m), ...) whose only lagged
// read of f is inside f's own update can fold one tighter than the read+write
// union footprint: fold factor = max lag m, not m+1. The write aliases the
// just-read (last-use) slot because the load happens before the store in the
// same statement. This is blocked when the lagged value has any *other* reader
// (another tuple component that survives split_tuples' let-hoisting, or a
// separate consumer func), which would need m+1.
//
// Build (against a libHalide with the StorageFolding change):
//   g++ fold_test.cpp -O2 -std=c++17 -I <build>/include -L <build>/src \
//       -lHalide -lpthread -ldl -o fold_test
//   LD_LIBRARY_PATH=<build>/src ./fold_test
//
// Expected with the change:
//   A lag-1 fold(1): PASS
//   B lag-2 fold(2): PASS
//   C tuple shared-lag fold(1): PASS
//   D external-lag fold(1): correctly REFUSED (error/abort) -- must NOT fold to 1.

#include "Halide.h"
#include <cstdio>
#include <vector>

using namespace Halide;

int main(int argc, char **argv) {
    // Which single case to run (so the "must-fail" case D can be run in its own
    // process). No arg => run A, B, C.
    std::string which = (argc > 1) ? argv[1] : "ABC";
    bool all = (which == "ABC");

    // ---- Case A: scalar lag-1 recurrence, fold_storage(i, 1) ----
    if (all || which == "A") {
        Func f(Int(32), "fA"), g("gA");
        Var i("i");
        f(i) = select(i <= 0, 0, likely(f(i - 1) + 1));
        g(i) = f(i);
        f.compute_at(g, i).store_root().fold_storage(i, 1);
        Buffer<int> out = g.realize({32});
        bool ok = true;
        for (int t = 0; t < 32; t++) ok &= (out(t) == t);
        printf("A lag-1 fold(1): %s\n", ok ? "PASS" : "FAIL");
    }

    // ---- Case B: scalar lag-2 recurrence, fold_storage(i, 2) ----
    if (all || which == "B") {
        Func f(Int(32), "fB"), g("gB");
        Var i("i");
        f(i) = select(i <= 1, i, likely(f(i - 2) + 1));
        g(i) = f(i);
        f.compute_at(g, i).store_root().fold_storage(i, 2);
        Buffer<int> out = g.realize({32});
        std::vector<int> ref(32);
        for (int t = 0; t < 32; t++) ref[t] = (t <= 1) ? t : ref[t - 2] + 1;
        bool ok = true;
        for (int t = 0; t < 32; t++) ok &= (out(t) == ref[t]);
        printf("B lag-2 fold(2): %s\n", ok ? "PASS" : "FAIL");
    }

    // ---- Case C: tuple, both components read f(i-1)[0] (shared lagged read
    // inside the SAME tuple update). fold_storage(i, 1). split_tuples hoists the
    // aliasing reads into lets, so factor 1 must be correct. ----
    if (all || which == "C") {
        Func f({Int(32), Int(32)}, "fC"), g("gC");
        Var i("i");
        f(i) = Tuple(select(i <= 0, 0, likely(f(i - 1)[0] + f(i - 1)[1])),
                     select(i <= 0, 1, likely(f(i - 1)[0])));
        g(i) = f(i)[0] + f(i)[1];
        f.compute_at(g, i).store_root().fold_storage(i, 1);
        Buffer<int> out = g.realize({32});
        std::vector<int> c0(32), c1(32);
        for (int t = 0; t < 32; t++) {
            c0[t] = (t <= 0) ? 0 : c0[t - 1] + c1[t - 1];
            c1[t] = (t <= 0) ? 1 : c0[t - 1];
        }
        bool ok = true;
        for (int t = 0; t < 32; t++) ok &= (out(t) == c0[t] + c1[t]);
        printf("C tuple shared-lag fold(1): %s\n", ok ? "PASS" : "FAIL");
    }

    // ---- Case D (safety): an EXTERNAL consumer reads f at a lag, so f(i-1) has
    // a second reader outside f's own update. The reduction must be REFUSED --
    // factor 1 is genuinely too small, so this should error/abort rather than
    // silently produce a wrong result. Run this case in its own process.
    if (which == "D") {
        Func f(Int(32), "fD"), g("gD");
        Var i("i");
        f(i) = select(i <= 0, 0, likely(f(i - 1) + 1));
        g(i) = f(i) + f(max(i - 1, 0));  // external lagged read of f
        f.compute_at(g, i).store_root().fold_storage(i, 1);
        Buffer<int> out = g.realize({32});
        // If we get here, folding did NOT refuse -- check whether it silently
        // corrupted (that would be a bug).
        std::vector<int> fv(32);
        for (int t = 0; t < 32; t++) fv[t] = (t <= 0) ? 0 : fv[t - 1] + 1;
        bool ok = true;
        for (int t = 0; t < 32; t++) ok &= (out(t) == fv[t] + fv[std::max(t - 1, 0)]);
        printf("D external-lag fold(1): %s (should have been refused!)\n",
               ok ? "ran-correct" : "SILENTLY-WRONG");
        return ok ? 0 : 1;
    }

    // ---- Case E (tuple safety): a tuple whose lagged value is read by an
    // EXTERNAL consumer. The lagged read of f.1 lives outside f's own update, so
    // the reduction must be REFUSED (factor 1 too small) rather than silently
    // corrupt. Run in its own process. ----
    if (which == "E") {
        Func f({Int(32), Int(32)}, "fE"), g("gE");
        Var i("i");
        f(i) = Tuple(select(i <= 0, 0, likely(f(i - 1)[0] + 1)),
                     select(i <= 0, 1, likely(f(i - 1)[1] + f(i - 1)[0])));
        g(i) = f(i)[0] + f(max(i - 1, 0))[1];  // external lagged read of f.1
        f.compute_at(g, i).store_root().fold_storage(i, 1);
        Buffer<int> out = g.realize({32});
        std::vector<int> c0(32), c1(32);
        for (int t = 0; t < 32; t++) {
            c0[t] = (t <= 0) ? 0 : c0[t - 1] + 1;
            c1[t] = (t <= 0) ? 1 : c1[t - 1] + c0[t - 1];
        }
        bool ok = true;
        for (int t = 0; t < 32; t++) ok &= (out(t) == c0[t] + c1[std::max(t - 1, 0)]);
        printf("E tuple external-lag fold(1): %s (should have been refused!)\n",
               ok ? "ran-correct" : "SILENTLY-WRONG");
        return ok ? 0 : 1;
    }

    printf("done\n");
    return 0;
}
