// Runner for the merge generator: builds two sorted arrays, merges them with
// the AOT-compiled pipeline, and checks the result against std::merge.

#include "HalideBuffer.h"
#include "merge.h"

#include <algorithm>
#include <cstdio>
#include <vector>

using namespace Halide::Runtime;

int main() {
    std::vector<int> a = {1, 3, 5, 7, 9, 11};
    std::vector<int> b = {2, 2, 4, 8, 8, 10, 12, 14};

    Buffer<int> A((int)a.size()), B((int)b.size()), out((int)(a.size() + b.size()));
    for (int t = 0; t < (int)a.size(); t++) A(t) = a[t];
    for (int t = 0; t < (int)b.size(); t++) B(t) = b[t];

    merge(A, B, (int)a.size(), (int)b.size(), out);

    std::vector<int> reference;
    std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(reference));

    for (int t = 0; t < out.width(); t++) {
        if (out(t) != reference[t]) {
            printf("Something went wrong!\n");
            return -1;
        }
    }

    printf("merged: ");
    for (int t = 0; t < out.width(); t++) printf("%d ", out(t));
    printf("\nSuccess!\n");
    return 0;
}
