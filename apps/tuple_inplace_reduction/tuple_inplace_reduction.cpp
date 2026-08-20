// Tuple-valued Func with an in-place reduction: component 0 is a running
// sum, component 1 is a running weighted sum whose update reads BOTH the
// just-updated component 0 and its own previous value.
//   f(x)[0] += in(r, x)
//   f(x)[1] += f(x)[0] * in(r, x)      -- depends on both 0 and 1
#include "Halide.h"
#include <cstdio>
using namespace Halide;

int main() {
    Var x("x");
    Buffer<float> in(8, 4);
    for (int xx = 0; xx < 4; xx++)
        for (int r = 0; r < 8; r++)
            in(r, xx) = (float)(r + 1) * (xx + 1);

    Func f(std::vector<Type>{Float(32), Float(32)}, "f");
    f(x) = Tuple(0.0f, 0.0f);

    RDom r(0, 8, "r");
    f(x) = Tuple(
        f(x)[0] + in(r, x),
        f(x)[1] + f(x)[0] * in(r, x));

    f.compute_root();

    Realization result = f.realize({4});
    Buffer<float> sum = result[0], wsum = result[1];
    for (int xx = 0; xx < 4; xx++)
        printf("x=%d sum=%f wsum=%f\n", xx, sum(xx), wsum(xx));
    return 0;
}
