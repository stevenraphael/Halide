// Halide tutorial lesson 25: Inductive functions

// This lesson demonstrates how to use Halide's inductive function feature to
// express schedules involving recursive functions that are otherwise difficult
// or impossible to express with RDoms.

// On linux, you can compile and run it like so:
// g++ lesson_25*.cpp -g -I <path/to/Halide.h -L <path/to/libHalide.so> -lHalide -lpthread -ldl -o lesson_25 -std=c++17
// LD_LIBRARY_PATH=<path/to/libHalide.so> ./lesson_25

// On os x:
// g++ lesson_25*.cpp -g -I <path/to/Halide.h -L <path/to/libHalide.so> -lHalide -o lesson_25 -std=c++17
// DYLD_LIBRARY_PATH=<path/to/libHalide.dylib> ./lesson_25

// If you have the entire Halide source tree, you can also build it by
// running:
//    make tutorial_lesson_25_inductive
// in a shell with the current directory at the top of the halide
// source tree.

#include "Halide.h"
#include "halide_benchmark.h"
#include <algorithm>
#include <cstdio>
#include <string>

using namespace Halide;
using namespace Halide::Tools; // for benchmark()

static int rowsum_calls = 0;

int count_rowsum_stores(JITUserContext *user_context, const halide_trace_event_t *e) {
    if (e->event == halide_trace_store && std::string(e->func) == "rowsum") {
        rowsum_calls++;
    }
    return 0;
}

int main() {
    // Declare some Vars to use below.
    Var x("x"), y("y"), c("c"), xo("xo"), yo("yo"), xi("xi"), yi("yi"), tile("tile");

    // If an RDom is used in a function, Halide forces the entire sequence of values in the RDom to be iterated over before
    // any of the values can be used in a subsequent function, thus preventing loop fusion (compute_at) where one of the loops
    // involves recursive iteration.
    // Inductive functions allow certain recursive functions to be expressed with pure definitions instead of updates with RDoms.
    // This enables loop fusion, unlocking optimizations that are impossible to express with RDoms.

    // Notes:
    // 1. All recursive calls must be guarded by a select() that divides the recursion into a base case and a recursive case.
    // 2. Functions can be declared with an explicit type. If a type is not specified, Halide will attempt to infer the type
    //    by analyzing the definition. Type inference can sometimes fail, so it is safest to explicitly provide a type.
    // 3. Inductive functions cannot be used as output buffers. They can only be used as intermediate functions in a pipeline.
    // 4. An inductive function is usually written as a single pure definition (as in the examples below), but it does not have
    //    to be. You can also define one with a non-inductive pure definition plus exactly one update definition, where the
    //    update definition is the inductive one. In that case, all RVars in the update definition must be nested inside all of
    //    the inductive variables.

    {
        // In this example we define a two-stage pipeline to compute a prefix mean. The first stage computes a prefix sum
        // of the input image, and the second stage divides the prefix sum by the number of elements.
        // First, we express this schedule using an RDom.
        // This is equivalent to the following C++ code:
        // for (int y = 0; y < 10; y++) {
        //     prefix_sum(0, y) = input(0, y);
        //     for (int x = 1; x < 10; x++) {
        //         prefix_sum(x, y) = prefix_sum(x - 1, y) + input(x, y);
        //     }
        //     for (int x = 0; x < 10; x++) {
        //         output(x, y) = prefix_sum(x, y) / (x + 1);
        //     }
        // }

        Func input("input"), prefix_sum("prefix_sum"), output("output");
        // The scan reads prefix_sum(r - 1, y), so r must start at 1; the base
        // case prefix_sum(0, y) is set by the update above.
        RDom r(1, 9);

        input(x, y) = x + y;
        prefix_sum(x, y) = undef<int>();
        prefix_sum(0, y) = input(0, y);
        prefix_sum(r, y) = prefix_sum(r - 1, y) + input(r, y);
        output(x, y) = prefix_sum(x, y) / (x + 1);

        prefix_sum.compute_at(output, y);

        printf("Pseudo-code for the RDom schedule:\n");
        output.print_loop_nest();
        printf("\n");

        // It should print:
        // produce output:
        //   for y:
        //     produce prefix_sum:
        //       prefix_sum(...) = ...
        //       for r4 in [1, 9]:
        //         prefix_sum(...) = ...
        //     consume prefix_sum:
        //       for x:
        //         output(...) = ...

        Buffer<int> result = output.realize({10, 10});

        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 10; x++) {
                int expected = (y * (x + 1) + x * (x + 1) / 2) / (x + 1);
                if (result(x, y) != expected) {
                    printf("Something went wrong!\n");
                    return -1;
                }
            }
        }
    }

    {
        // Next, we express the same unoptimized schedule using an inductive definition for prefix_sum.
        // The select() function is required in an inductive definition: all recursive calls
        // must be guarded by a select() that divides the recursion into a base case and a
        // recursive case. The likely() function is used to indicate that the recursive case is
        // a steady state, which can help the compiler by triggering loop partitioning
        // and eliminating the select() operation.

        Func input("input"), prefix_sum(Int(32), "prefix_sum"), output("output");
        input(x, y) = x + y;
        prefix_sum(x, y) = select(x <= 0, input(0, y), likely(prefix_sum(x - 1, y) + input(x, y)));
        output(x, y) = prefix_sum(x, y) / (x + 1);

        prefix_sum.compute_at(output, y);

        Buffer<int> result = output.realize({10, 10});

        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 10; x++) {
                int expected = (y * (x + 1) + x * (x + 1) / 2) / (x + 1);
                if (result(x, y) != expected) {
                    printf("Something went wrong!\n");
                    return -1;
                }
            }
        }
    }

    {
        // We express an optimized schedule for the same algorithm definition using inductive functions.
        // By removing prefix_sum.compute_at(output, y), we compute a single value of prefix_sum immediately
        // before using it in output, rather than computing an entire row of prefix_sum beforehand.
        // By adding prefix_sum.store_at(output, y), we store the values of prefix_sum in a buffer that is
        // preserved across the computation of a single row of output, which prevents having to recompute
        // the entire row for each value of output. Since the algorithm only ever reads the single most
        // recent value of prefix_sum, we can add prefix_sum.fold_storage(x, 1) to reduce the storage to a
        // single element.
        //
        // This schedule is equivalent to the following C++ code:
        // for (int y = 0; y < 10; y++) {
        //     int prefix_sum;
        //     prefix_sum = input(0, y);
        //     for (int x = 1; x < 10; x++) {
        //         prefix_sum = prefix_sum + input(x, y);
        //         output(x, y) = prefix_sum / (x + 1);
        //     }
        // }
        // This schedule uses a single loop to compute two functions, prefix_sum and output, which is not possible with an RDom.
        // In addition, this schedule only requires a buffer of size 1 for prefix_sum, which the compiler transforms into a register.
        //
        // We could do this without inductive functions by using a tuple-valued function to compute prefix_sum and output
        // in a single stage, but this function would wastefully output two values per pixel, both of which are stored in memory.

        Func input("input"), prefix_sum(Int(32), "prefix_sum"), output("output");
        input(x, y) = x + y;
        prefix_sum(x, y) = select(x <= 0, input(0, y), likely(prefix_sum(x - 1, y) + input(x, y)));
        output(x, y) = prefix_sum(x, y) / (x + 1);

        prefix_sum.compute_at(output, x).store_at(output, y).fold_storage(x, 1);

        printf("Loop nest for the inductive schedule:\n");
        output.print_loop_nest();
        printf("\n");

        // It should print (the "x." loop is the sliding window over x, and the
        // inner "for x" is its short warm-up region):
        // produce output:
        //   for y:
        //     store prefix_sum:
        //       for x.:
        //         produce prefix_sum:
        //           for x:
        //             prefix_sum(...) = ...
        //         consume prefix_sum:
        //           output(...) = ...

        Buffer<int> result = output.realize({10, 10});

        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 10; x++) {
                int expected = (y * (x + 1) + x * (x + 1) / 2) / (x + 1);
                if (result(x, y) != expected) {
                    printf("Something went wrong!\n");
                    return -1;
                }
            }
        }
    }

    {
        // Demonstrate a chained inductive function schedule.
        // A summed-area table is built from two prefix sums:
        // first along x, then along y. Each is an inductive function.
        // We then output a box filter that sums the input
        // over a (2R+1)x(2R+1) window using the table.
        // We also count the number of times the rowsum function is computed to verify that it is only computed once per pixel.

        const int W = 10, H = 10, R = 4;

        Func input("input");
        input(x, y) = x + y;

        Func rowsum = Func(Int(32), "rowsum");
        rowsum(x, y) = select(x <= 0, input(0, y),
                            likely(rowsum(x - 1, y) + input(x, y)));

        Func sat = Func(Int(32), "sat");
        sat(x, y) = select(y < 0, 0,
                        likely(sat(x, y - 1) + rowsum(x, y)));

        Func sat_clamped("sat_clamped");
        sat_clamped(x, y) = select(x < 0 || y < 0, 0,
                                sat(clamp(x, 0, W - 1), clamp(y, 0, H - 1)));

        Func box("box");
        box(x, y) = sat_clamped(x + R, y + R)
                - sat_clamped(x - R - 1, y + R)
                - sat_clamped(x + R, y - R - 1)
                + sat_clamped(x - R - 1, y - R - 1);

        box.bound(x, 0, W).bound(y, 0, H);

        sat.store_root().compute_at(box, y).fold_storage(y, 2 * R + 2);
        rowsum.store_at(box, y).compute_at(sat, x).fold_storage(x, 1);

        rowsum.trace_stores();
        box.jit_handlers().custom_trace = &count_rowsum_stores;

        Buffer<int> result = box.realize({W, H});

        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int expected = 0;
                for (int j = std::max(0, y - R); j <= std::min(H - 1, y + R); j++) {
                    for (int i = std::max(0, x - R); i <= std::min(W - 1, x + R); i++) {
                        expected += i + j;
                    }
                }
                if (result(x, y) != expected) {
                    printf("Something went wrong!\n");
                    return -1;
                }
            }
        }

        printf("rowsum was computed %d times\n", rowsum_calls);
        if (rowsum_calls != W * H) {
            printf("Something went wrong!\n");
            return -1;
        }
    }

    printf("Success!\n");
    return 0;
}
