#include "Halide.h"
#include "check_call_graphs.h"
#include "test_sharding.h"

#include <cstdio>
#include <map>

namespace {

using std::map;
using std::string;

using namespace Halide;
using namespace Halide::Internal;

int simple_inductive_test() {
    Func g("g");
    Var x("x"), y("y");

    // g(x, y) = x + y;
    // g(r.x, r.y) = g(r.x, r.y);
    g(x, y) = select(x <= 0, 0, g(max(0, x - 1), y) + x + y);

    Buffer<int> im = g.realize({600, 5});
    auto func = [](int x, int y) {
        return y * x + x * (x + 1) / 2;
    };
    if (check_image(im, func)) {
        return 1;
    }
    return 0;
}

int reorder_test() {
    Func g("g"), h("h");
    Var x("x"), y("y");

    Var xi("xi"), xii("xii"), xo("xo");

    // g(x, y) = x + y;
    // g(r.x, r.y) = g(r.x, r.y);
    g(x, y) = select(x <= 0, 0, g(max(0, x - 1), y) + x + y);

    h(x, y) = g(x + 5, y) / 4;
    h.split(x, xo, xi, 24).reorder(xi, y, xo);

    g.compute_at(h, xo).store_root();

    g.split(x, xi, xii, 5).reorder(xii, y, xi).vectorize(y, 8);

    Buffer<int> im = h.realize({80, 80});
    auto func = [](int x, int y) {
        return (y * (x + 5) + (x + 5) * (x + 6) / 2) / 4;
    };
    if (check_image(im, func)) {
        return 1;
    }
    return 0;
}

int summed_area_table() {
    Func f("f"), g("g"), h("h");
    Var x("x"), y("y");
    f(x, y) = x + y;
    g(x, y) = f(x, y) + select(x <= 0, 0, g(x - 1, y)) + select(y <= 0, 0, g(x, y - 1)) - select(x <= 0 || y <= 0, 0, g(x - 1, y - 1));
    h(x, y) = g(x, y) / 8;
    g.compute_at(h, x).store_root();

    Buffer<int> im = h.realize({80, 80});
    auto func = [](int x, int y) {
        return (x * (x + 1) / 2 * (y + 1) + y * (y + 1) / 2 * (x + 1)) / 8;
    };
    if (check_image(im, func)) {
        return 1;
    }
    return 0;
}

int large_baseline() {
    Func g("g"), h("h");
    Var x("x"), y("y");

    // g(x, y) = x + y;
    // g(r.x, r.y) = g(r.x, r.y);
    g(x, y) = select(x <= 8, (y * x + x * (x + 1) / 2) - 1, g(x - 1, y) + x + y);

    h(x, y) = g(x + 5, y) / 4;

    g.compute_at(h, x).store_at(h, y);

    Buffer<int> im = h.realize({80, 80});
    auto func = [](int x, int y) {
        return (y * (x + 5) + (x + 5) * (x + 6) / 2 - 1) / 4;
    };
    if (check_image(im, func)) {
        return 1;
    }
    return 0;
}

int fibonacci() {
    Func g("g"), h("h");
    Var x("x"), y("y");

    // g(x, y) = x + y;
    // g(r.x, r.y) = g(r.x, r.y);
    g(x, y) = select(x <= 1, 1, g(x - 1, y) + g(x - 2, y));
    h(x, y) = g(x, y);

    h.bound(x, 0, 80);
    Buffer<int> im = h.realize({80, 80});
    auto func = [](int x, int y) {
        int a = 1;
        int b = 1;
        for (int i = 2; i <= x; i++) {
            int c = a + b;
            b = a;
            a = c;
        }
        return a;
    };
    if (check_image(im, func)) {
        return 1;
    }
    return 0;
}

int sum_2d_test() {
    Func f("f"), g("g"), h("h");
    Var x("x"), y("y");
    f(x, y) = select(x <= 0, 0, x + f(x - 1, y));
    g(x, y) = select(y <= 0, f(x, 0), f(x, y) + g(x, y - 1));
    h(x, y) = g(x, y);
    h.bound(x, 0, 80).bound(y, 0, 80).vectorize(x, 8);
    g.compute_at(h, x).store_root().vectorize(x, 8);
    f.compute_at(h, x).store_root();
    Buffer<int> im = h.realize({80, 80});
    auto func = [](int x, int y) {
        int ans = 0;
        for (int a = 0; a <= x; a++) {
            for (int b = 0; b <= y; b++) {
                ans += a;
            }
        }
        return ans;
    };
    if (check_image(im, func)) {
        return 1;
    }
    return 0;
}

int sum_1d_test() {
    Func f("f"), g("g"), h("h");
    Var x("x"), y("y");
    f(x, y) = x + y;
    f(x, y) += x;  // select(x<=0, 0, x+f(x-1,y));
    g(x, y) = select(y <= 0, f(x, 0), f(x, y) + g(x, y - 1));
    h(x, y) = g(x, y);
    h.bound(x, 0, 80).bound(y, 0, 80);
    // stress-testing bounds inference for dependent non-inlined funcs
    f.compute_at(h, x);
    Buffer<int> im = h.realize({80, 80});
    auto func = [](int x, int y) {
        int ans = 0;
        for (int a = 0; a <= y; a++) {
            ans += 2 * x + a;
        }
        return ans;
    };
    if (check_image(im, func)) {
        return 1;
    }
    return 0;
}

int multi_baseline_test() {
    Func f("f"), g("g"), h("h");
    Var x("x"), y("y");
    f(x, y) = x + y;
    f(x, y) += x;  // select(x<=0, 0, x+f(x-1,y));
    g(x, y) = select(y <= 0, f(x, 0), f(x, y) + g(x, y - 1)) + select(y <= 3, f(x, 0), f(x, y) + g(x, y - 1));
    h(x, y) = g(x, y);
    h.bound(x, 0, 80).bound(y, 0, 20);
    f.compute_at(h, x);
    Buffer<int> im = h.realize({80, 20});
    auto func = [](int x, int y) {
        std::vector<int> ans;

        for (int a = 0; a <= y; a++) {
            if (a <= 0) {
                ans.emplace_back(4 * x);
            } else if (a <= 3) {
                ans.emplace_back(2 * x + (2 * x + a) + ans[a - 1]);
            } else {
                ans.emplace_back(2 * x + a + ans[a - 1] + (2 * x + a) + ans[a - 1]);
            }
        }
        return ans[y];
    };
    if (check_image(im, func)) {
        return 1;
    }
    return 0;
}

int type_declare_test() {
    Func g = Func(Int(32), "g");
    Func h("h");
    Var x("x"), y("y");

    g(x, y) = select(x <= 0, 0, 1 + g(max(0, x - 1), y) + x + 2);

    h(x, y) = g(x + 5, y) / 4;

    g.compute_at(h, x).store_at(h, y);

    Buffer<int> im = h.realize({600, 5});
    auto func = [](int x, int y) {
        return (3 * (x + 5) + (x + 5) * (x + 6) / 2) / 4;
    };
    if (check_image(im, func)) {
        return 1;
    }
    return 0;
}

int blur_test(){
    Func f1("f1"), f2("f2"), f3("f3"), f4("f4"), f5("f5");
    Var x;
    f1(x) = x*x+1;
    f2(x) = select(x <= 0, 0, f1(x) + f2(x-1));
    f3(x) = select(x <= 0, 0, f2(x) + f3(x-1));
    f4(x) = select(x <= 0, 0, f3(x) + f4(x-1));
    f5(x) = f4(x);
    f2.store_root().compute_at(f5, x).fold_storage(x,2);
    f3.store_root().compute_at(f5, x).fold_storage(x,2);
    f4.store_root().compute_at(f5, x).fold_storage(x,2);
    //f4.store_root().compute_root();
    f5.compile_to_lowered_stmt("blurnr.txt", {}, Halide::Text);
    Buffer<int> im = f5.realize({100});
    return 0;

}

int storage_test(){
    Func f1("f1"), f2("f2");
    Var x;
    f1(x) = select(x <= 0, 0, f1(x-1) + x);
    f2(x) = f1(x) + 1;
    f1.store_root().compute_at(f2, x);
    f2.compile_to_lowered_stmt("storatagenr.txt", {}, Halide::Text);
    Buffer<int> im = f2.realize({100});
    return 0;
}

int blur2_test(){
    Func f1("f1"), f2("f2"), f3("f3"), f4("f4"), f5("f5");
    Var x;
    f1(x) = x*x+1;
    f2(x) = select(x <= 0, 0, f1(x) + f2(x-1));
    f3(x) = select(x <= 0, 0, f2(x) + f3(x-1));
    //f4(x) = select(x <= 0, 0, f3(x) + f4(x-1));
    f4(x) = f3(x);
    //f1.store_root().compute_at(f2, x);
    f2.store_root().compute_at(f3, x);
    f3.store_root().compute_root();//(f4, x);
    //f4.store_root().compute_root();
    f4.compile_to_lowered_stmt("blur2.txt", {}, Halide::Text);
    Buffer<int> im = f4.realize({100});
    return 0;

}

int parallel_test(){
    Func f1("f1"), f2("f2"), f3("f3");
    Var x("x"), y("y");
    f1(x, y) = x * x + y + 1;
    f2(x, y) = select(x <= 0||y<=0, 0, f1(x, y) + f2(x-1, y-1));
    f3(x, y) = f2(x, y);
    f2.store_root().compute_at(f3, y).parallel(x);
    Buffer<int> im = f3.realize({100, 100});

    Func g1("g1");
    RDom r(1, 99, 1, 99);
    g1(x, y) = 0;
    g1(r.x, r.y) += g1(r.x - 1, r.y - 1) + r.x * r.x + r.y + 1;
    Buffer<int> im2 = g1.realize({100, 100});
    auto func = [&im2](int x, int y) {
        return im2(x, y);
    };
    if(check_image(im, func)){
        return 1;
    }
    return 0;
}

int parallel_test_2(){
    Func f1("f1"), f2("f2"), f3("f3");
    Var x("x"), y("y"), z("z");
    f1(x, y,z) = x * x + y*z + 1;
    f2(x, y,z) = select(x <= 0||y<=0||z<=0, 0, f1(x, y,z) + f2(x, y-1,z)+f2(x-1,y,z)+f2(x-1,y,z-1));
    f3(x, y,z) = f2(x, y,z);
    f3.reorder(z,x,y);
    f2.compute_root().reorder(z,x,y).parallel(z);
    Buffer<int> im = f3.realize({20, 20, 20});
    Func g1("g1");
    RDom r(1, 19, 1, 19, 1, 19);
    g1(x, y, z) = 0;
    g1(r.x, r.y, r.z) += g1(r.x, r.y - 1, r.z) + g1(r.x - 1, r.y, r.z) + g1(r.x - 1, r.y, r.z - 1) + r.x * r.x + r.y * r.z + 1;
    Buffer<int> im2 = g1.realize({20, 20, 20});
    auto func = [&im2](int x, int y, int z) {   
        return im2(x, y, z);
    };
    if(check_image(im, func)){
        return 1;
    }
    return 0;
}

int parallel_test_3(){
    Func f1("f1"), f2("f2"), f3("f3");
    Var x("x"), y("y"), z("z"), xo("xo"), xi("xi"), zo("zo"), zi("zi");
    f1(x, y,z) = x * x + y*z + 1;
    f2(x, y,z) = select(x <= 0||y<=0||z<=0, 0, f1(x, y,z) + f2(x, y-1,z)+f2(x-1,y,z)+f2(x-1,y,z-1));
    f3(x, y,z) = f2(x, y,z);
    f3.reorder(z,x,y);
    f2.split(x,xo,xi,8).split(z,zo,zi,8);
    f2.compute_root().reorder(zo, zi, xi, xo, y).parallel(zi); 
    Buffer<int> im = f3.realize({20, 20, 20});
    Func g1("g1");
    RDom r(1, 19, 1, 19, 1, 19);
    g1(x, y, z) = 0;
    g1(r.x, r.y, r.z) += g1(r.x, r.y - 1, r.z) + g1(r.x - 1, r.y, r.z) + g1(r.x - 1, r.y, r.z - 1) + r.x * r.x + r.y * r.z + 1;
    Buffer<int> im2 = g1.realize({20, 20, 20});
    auto func = [&im2](int x, int y, int z) {   
        return im2(x, y, z);
    };
    if(check_image(im, func)){
        return 1;
    }
    return 0;
}

int parallel_test_4(){
    Func f1("f1"), f2("f2"), f3("f3");
    Var x("x"), xo("xo"), xi("xi");
    f1(x) = select(x<=0, 0, f1(x-8) + f1(x-16)+x*x);
    f2(x) = f1(x);
    f1.compute_root().vectorize(x, 9);//split(x, xo, xi, 8).reorder(xo, xi).parallel(xi);
    Buffer<int> im = f2.realize({100});
    return 0;
}

int sum_scan() {
    Func f4 = Func(Int(32), "f4");
    Func f1("f1"), f2("f2"), f3("f3"), f5("f5"), in("in");
    Var x("x"), xo("xo"), xi("xi");
    in(x) = x + 1;
    f1(x) = in(x) + in(x + 4);
    f2(x) = f1(x) + f1(x + 2);
    f3(x) = f2(x) + f2(x + 1);
    f4(x) = select(x < 8, f3(x-8)-f3(x-8), likely(f3(x-8) + f4(x-8)));
    f5(x) = f4(x) / 36;
    f5.bound(x, 0, 1024).split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    //f4.bound(x, -8, 1024);
    //f3.bound(x, 0, 1025);
    //f2.bound(x, 0, 1026);
    //f1.bound(x, 0, 1028);
    
    //f5.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    f4.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi).store_root().compute_at(f5, xo);
    for (Func f :  {f1, f2, f3}) {
        f.compute_at(f5, xo).vectorize(x).store_in(MemoryType::Register).fold_storage(x, 24);
    }
    f3.fold_storage(x, 8);
    f4.fold_storage(x, 16);
    f5.compile_to_lowered_stmt("sumnr.txt", {}, Halide::Text);
    
    return 0;
}

int bad_check(){
    Func f1("f1"), f2("f2");
    Var x("x");
    f1(x) = select(x<=0 || x > 10, 0, f1(x-1)+f1(x+1)+x);
    f2(x) = f1(x);
    Buffer<int> im = f2.realize({20});
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    struct Task {
        std::string desc;
        std::function<int()> fn;
    };

    std::vector<Task> tasks = {
        //{"parallel test 2", parallel_test_4},
        /*{"parallel test", parallel_test},
        {"parallel test 2", parallel_test_2},
        {"parallel test 3", parallel_test_3},
        {"simple inductive test", simple_inductive_test},
        {"reordering test", reorder_test},
        {"summed area table test", summed_area_table},
        {"large baseline test", large_baseline},
        {"fibonacci test", fibonacci},
        {"2d sum test", sum_2d_test},
        {"1d sum test", sum_1d_test},
        {"multi-baseline test", multi_baseline_test},
        {"type declaration test", type_declare_test},
        {"blur test", blur_test},*/
        //{"blur2 test", blur_test},
        {"sum scan test", sum_scan},
        //{"bad check test", bad_check},
        //{"storage test", storage_test},
    };

    using Sharder = Halide::Internal::Test::Sharder;
    Sharder sharder;
    for (size_t t = 0; t < tasks.size(); t++) {
        if (!sharder.should_run(t)) continue;
        const auto &task = tasks.at(t);
        std::cout << task.desc << "\n";
        if (task.fn() != 0) {
            return 1;
        }
    }

    printf("Success!\n");
    return 0;
}
