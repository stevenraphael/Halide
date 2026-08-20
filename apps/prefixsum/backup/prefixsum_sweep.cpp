// Sweep W (and H) for the RDom-vs-inductive prefix-sum comparison from
// tutorial/lesson_25_inductive.cpp, to see how the two schedules trend as
// the per-row working set grows relative to cache.
#include "Halide.h"
#include "halide_benchmark.h"
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace Halide;
using namespace Halide::Tools;

double time_rdom(int W, int H) {
    Var x("x"), y("y");
    Func in_a("in_a"), prefix_sum_a("prefix_sum_a"), output_a("output_a");
    RDom r(1, W - 1);
    in_a(x, y) = x + y;
    prefix_sum_a(x, y) = undef<int>();
    prefix_sum_a(0, y) = in_a(0, y);
    prefix_sum_a(r, y) = prefix_sum_a(r - 1, y) + in_a(r, y);
    output_a(x, y) = prefix_sum_a(x, y) / (x + 1);
    prefix_sum_a.compute_at(output_a, y);
    output_a.compile_jit();
    Buffer<int> result_a(W, H);
    return benchmark(10, 10, [&]() { output_a.realize(result_a); });
}

double time_inductive(int W, int H, int fold) {
    Var x("x"), y("y");
    Func in_b("in_b"), prefix_sum_b(Int(32), "prefix_sum_b"), output_b("output_b");
    in_b(x, y) = x + y;
    prefix_sum_b(x, y) = select(x <= 0, in_b(0, y), likely(prefix_sum_b(x - 1, y) + in_b(x, y)));
    output_b(x, y) = prefix_sum_b(x, y) / (x + 1);
    prefix_sum_b.compute_at(output_b, x).store_at(output_b, y).fold_storage(x, fold);
    output_b.bound(x, 0, W).bound(y, 0, H);
    output_b.compile_jit();
    Buffer<int> result_b(W, H);
    return benchmark(10, 10, [&]() { output_b.realize(result_b); });
}

int main(int argc, char **argv) {
    int argi = 1;
    bool reverse_order = argc > argi && std::string(argv[argi]) == "reverse";
    if (reverse_order) argi++;

    int H = argc > argi ? atoi(argv[argi++]) : 32;
    std::vector<int> Ws = {65536, 262144, 1048576, 4194304, 16777216};
    if (argc > argi) {
        Ws.clear();
        for (; argi < argc; argi++) Ws.push_back(atoi(argv[argi]));
    }

    printf("%-12s %-10s %-10s %-10s %-8s\n", "W", "RDom(ms)", "ind1(ms)", "ind2(ms)", "speedup");
    for (int W : Ws) {
        double t_rdom, t_ind1, t_ind2;
        if (reverse_order) {
            t_ind1 = time_inductive(W, H, 1);
            t_ind2 = time_inductive(W, H, 2);
            t_rdom = time_rdom(W, H);
        } else {
            t_rdom = time_rdom(W, H);
            t_ind1 = time_inductive(W, H, 1);
            t_ind2 = time_inductive(W, H, 2);
        }
        printf("%-12d %-10.3f %-10.3f %-10.3f %.3fx\n",
               W, t_rdom * 1e3, t_ind1 * 1e3, t_ind2 * 1e3, t_rdom / t_ind1);
    }
    return 0;
}
