// This file defines a generator that merges two sorted 1-D arrays into one
// sorted array using an inductive Halide function. Merging is a sequential
// process: the number of elements consumed from each input so far depends on
// all the comparisons made before it, which is exactly the kind of recursion an
// inductive function expresses. See tutorial/lesson_25_inductive.cpp for an
// introduction to inductive funcs.

#include "Halide.h"
#include <vector>

using namespace Halide;
using std::vector;

Func merge_unsorted(Func A, int nA, int nstages, Var k) {
    vector<Func> stages;
    int cursize = 1;
    for(int i = 0; i < nstages; i++) {
        Func stage(Int(32), "stage" + std::to_string(i));
        Func istage(Int(32), "istage" + std::to_string(i));
        Var j("j"), x("x");
        Expr ip = istage(j - 1, x);
        Expr jp = (j - 1) - ip;
        Expr tookA;
        if (i == 0) {
            // Stage 0 merges the adjacent single elements A[2x] and A[2x+1].
            tookA = (ip < cursize) && ((jp >= cursize) || (A(2*x + ip) <= A(2*x + 1 + jp)));
        } else {
            Expr ipc = unsafe_promise_clamped(ip, 0, cursize - 1), jpc = unsafe_promise_clamped(jp, 0, cursize - 1);
            tookA = (ip < cursize) && ((jp >= cursize) || (stages[i-1](ipc,2*x) <= stages[i-1](jpc,2*x+1)));
        }
        istage(j, x) = select(j <= 0, 0, likely(ip + select(tookA, 1, 0)));
        Expr ik = istage(j, x), jk = j - istage(j, x);
        Expr takeA;
        if (i == 0) {
            takeA = (ik < cursize) && ((jk >= cursize) || (A(2*x + ik) <= A(2*x + 1 + jk)));
            stage(j, x) = select(takeA, A(2*x + ik), A(2*x + 1 + jk));
        } else {
            Expr ikc = unsafe_promise_clamped(ik, 0, cursize - 1), jkc = unsafe_promise_clamped(jk, 0, cursize - 1);
            takeA = (ik < cursize) && ((jk >= cursize) || (stages[i-1](ikc,2*x) <= stages[i-1](jkc,2*x+1)));
            stage(j, x) = select(takeA, stages[i-1](ikc,2*x), stages[i-1](jkc,2*x+1));
        }
        stage.compute_root();
        istage.compute_at(stage, j).store_root();
        cursize *= 2;
        stages.push_back(stage);
    }
    Func output("output");
    output(k) = stages.back()(k, 0);
    return output;
}

// Build a Func that merges two sorted 1-D sequences A (length nA) and B
// (length nB) into a single sorted sequence of length nA + nB.
Func merge_sorted(Func A, Func B, int nA, int nB, Var k) {
    // i(k) is the number of elements taken from A before output element k is
    // produced (so k - i(k) have been taken from B). To decide element k we
    // compare the next unconsumed element of each array, taking from A unless A
    // is exhausted or B's next element is smaller. This depends on the previous
    // state, so i is an inductive function: the base case is i(0) = 0 and the
    // recursive case advances the count by one comparison.
    Func i = Func(Int(32), "i");
    Expr ip = i(k - 1);
    Expr jp = (k - 1) - ip;
    Expr tookA = (ip < nA) && ((jp >= nB) || (A(ip) <= B(jp)));
    i(k) = select(k <= 0, 0, likely(ip + select(tookA, 1, 0)));

    // Emit the element chosen at step k using the current counts.
    Func merged("merged");
    Expr ik = i(k), jk = k - i(k);
    Expr takeA = (ik < nA) && ((jk >= nB) || (A(ik) <= B(jk)));
    merged(k) = select(takeA, A(ik), B(jk));

    // Compute one value of the inductive counter just before it is used, keeping
    // the running state live across the output loop.
    i.compute_at(merged, k).store_root();
    return merged;
}

class Merge : public Generator<Merge> {
public:
    Input<Buffer<int, 1>> A{"A"};
    Input<Buffer<int, 1>> B{"B"};
    Input<int> nA{"nA"};
    Input<int> nB{"nB"};

    Output<Buffer<int, 1>> merged{"merged"};

    void generate() {
        Var k("k");

        // Clamp the array accesses so speculative reads at the boundary stay in
        // bounds; the merge logic never uses an out-of-range value.
        Func Af("Af"), Bf("Bf");
        Af(k) = A(clamp(k, 0, max(nA - 1, 0)));
        Bf(k) = B(clamp(k, 0, max(nB - 1, 0)));

        merged = merge_sorted(Af, Bf, nA, nB, k);
    }
};

HALIDE_REGISTER_GENERATOR(Merge, merge)
