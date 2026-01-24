#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"

#include <iostream>

using namespace Halide;
using namespace Halide::Tools;

void fill_buffer_a_f32(Buffer<float> &buf, int row, int acc, int ch) {
    for (int iy = 0; iy < row; ++iy) {
        for (int ix = 0; ix < acc; ++ix) {
            for(int iz = 0; iz < ch; ++iz) {
                // value between 0 and 100
                float val = float(((float)rand() / (float)(RAND_MAX)) * 100.f);
                buf(ix, iy, iz) = val;
            }
        }
    }
}

Func sum_scan_inductive(Func input, Expr length) {
    Func f4 = Func(Int(32), "f4");
    Func f1("f1"), f2("f2"), f3("f3"), f5("f5"), in("in");
    Var x("x"), xo("xo"), xi("xi");
    in(x) = x + 1;
    f1(x) = input(x) + input(x + 4);
    f2(x) = f1(x) + f1(x + 2);
    f3(x) = f2(x) + f2(x + 1);
    f4(x) = select(x < 8, 0, likely(f3(x) + f4(x-8)));
    f5(x) = f4(x) / 36;
    f5.bound(x, 0, length).split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    //f4.bound(x, -8, 1024);
    //f3.bound(x, 0, 1025);
    //f2.bound(x, 0, 1026);
    //f1.bound(x, 0, 1028);
    
    //f5.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    f4.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi).store_root().compute_at(f5, xo);
    for (Func f :  {f1, f2, f3}) {
        f.compute_at(f5, xo).vectorize(x).store_in(MemoryType::Register);//.fold_storage(x, 24);
    }
    //f3.fold_storage(x, 8);
    //f4.fold_storage(x, 16);

    return f5;
}


Func sum_scan_bad(Func input, Expr length) {
    Func f4 = Func(Int(32), "f4");
    Func f1("f1"), f2("f2"), f3("f3"), f5("f5"), in("in");
    Var x("x"), xo("xo"), xi("xi");
    in(x) = x + 1;
    f1(x) = input(x) + input(x + 4);
    f2(x) = f1(x) + f1(x + 2);
    f3(x) = f2(x) + f2(x + 1);
    f4(x) = select(x < 8, 0, f3(x));
    f5(x) = f4(x) / 36;
    f5.bound(x, 0, length).split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    //f4.bound(x, -8, 1024);
    //f3.bound(x, 0, 1025);
    //f2.bound(x, 0, 1026);
    //f1.bound(x, 0, 1028);
    
    //f5.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    f4.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi).store_root().compute_at(f5, xo);
    for (Func f :  {f1, f2, f3}) {
        f.compute_at(f5, xo).vectorize(x).store_in(MemoryType::Register);//.fold_storage(x, 24);
    }
    //f3.fold_storage(x, 8);
    //f4.fold_storage(x, 16);

    return f5;
}


Func sum_scan_normal(Func input, Expr length) {
    Func f4 = Func(Int(32), "f4");
    Func f1("f1"), f2("f2"), f3("f3"), f5("f5"), in("in");
    Var x("x"), xo("xo"), xi("xi");
    in(x) = x + 1;
    f1(x) = input(x) + input(x + 4);
    f2(x) = f1(x) + f1(x + 2);
    f3(x) = f2(x) + f2(x + 1);
    f4(x) = select(x < 8, f3(x)-f3(x), likely(f3(x) + f4(x-8)));
    f5(x) = f4(x) / 36;
    f5.bound(x, 0, length).split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    //f4.bound(x, -8, 1024);
    //f3.bound(x, 0, 1025);
    //f2.bound(x, 0, 1026);
    //f1.bound(x, 0, 1028);
    
    //f5.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    f4.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi).compute_root();
    for (Func f :  {f1, f2, f3}) {
        f.compute_at(f4, xo).vectorize(x).store_in(MemoryType::Register).fold_storage(x, 24);
    }
    //f3.fold_storage(x, 8);
    //f4.fold_storage(x, 16);

    return f5;
}

Func blur_cols_inductive(Func input, Expr height, Expr alpha) {
    Var x, y, c;

    const int vec = 8;
    

    // Pure definition: do nothing.
    Func blur = Func(Float(32), "blur");
    blur(x, y, c) = input(x,y,c);//select(y <= 0, input(x, y, c), likely((1 - alpha) * blur(x, y - 1, c) + alpha * input(x, y, c)));
    
    Func blur2 = Func(Float(32), "blur2");
    blur2(x, y, c) = blur(x,y,c);//select(y <= 0, blur(x, height - 1, c), likely((1 - alpha) * blur2(x, y - 1, c) + alpha * blur(x, height - y - 1, c)));

    Func transpose("transpose");
    transpose(x, y, c) = undef<float>();

    RDom rx(0, height);
    
    transpose(height-rx-1, y, c) = blur2(y,rx, c);

    Var xo, yo,xi,yi, t;
    transpose.bound(y, 0, height);
    transpose.bound(x, 0, height);
    RVar rxi, rxo;
    transpose.compute_root().update(0)
        .tile(rx, y, rxo, yo, rxi, yi, 8, vec)
        //.parallel(yo)
        //.parallel(c)
        //.vectorize(rxi)
        ;
    blur.compute_at(transpose, yo).reorder(x, y);//.vectorize(x);
    blur2.compute_at(transpose, rxo).store_at(transpose, yo).reorder(x, y).store_in(MemoryType::Register);//.vectorize(x).fold_storage(y, 16);
    //
    return transpose;
}

Func blur_cols_transpose(Func input, Expr height, Expr alpha) {
    Var x, y, c;
    Func blur("blur");

    const int vec = 8;

    // Pure definition: do nothing.
    blur(x, y, c) = undef<float>();
    // Update 0: set the top row of the result to the input.
    blur(x, 0, c) = input(x, 0, c);
    // Update 1: run the IIR filter down the columns.
    RDom ry(1, height - 1);
    blur(x, ry, c) =
        (1 - alpha) * blur(x, ry - 1, c) + alpha * input(x, ry, c);
    // Update 2: run the IIR blur up the columns.
    Expr flip_ry = height - ry - 1;
    blur(x, flip_ry, c) =
        (1 - alpha) * blur(x, flip_ry + 1, c) + alpha * blur(x, flip_ry, c);

    // Transpose the blur.
    Func transpose("transpose");
    transpose(x, y, c) = blur(y, x, c);

    // Schedule

    // CPU schedule.
    // 8.2ms on an Intel i9-9960X using 16 threads
    // Split the transpose into tiles of rows. Parallelize over channels
    // and strips (Halide supports nested parallelism).
    Var xo, yo, t;
    transpose.compute_root()
        .tile(x, y, xo, yo, x, y, vec, vec * 4)
        //.vectorize(x)
        //.parallel(yo)
        //.parallel(c)
        ;

    // Run the filter on each row of tiles (which corresponds to a strip of
    // columns in the input).
    blur.compute_at(transpose, yo);

    //blur.reorder(x,y).vectorize(x);

    // Vectorize computations within the strips.
    blur.update(0)
        .unscheduled();
    blur.update(1)
        .reorder(x, ry)
        //.vectorize(x)
        ;
    blur.update(2)
        .reorder(x, ry)
        //.vectorize(x)
        ;
        
    return transpose;
}

bool iir_normal() {
    const int row = 2016;
    const int acc = 2016;
    const int ch = 3;

    ImageParam A(Float(32), 3, "input");

    Func result = blur_cols_transpose(Func(A), row, 0.1f);

    Buffer<float> a_buf(acc, row, ch);
    fill_buffer_a_f32(a_buf, row, acc, ch);
    A.set(a_buf);

    Buffer<float> out(row, acc, ch);
    auto time = Tools::benchmark(20, 20, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

bool iir_inductive() {
    const int row = 2016;
    const int acc = 2016;
    const int ch = 3;

    ImageParam A(Float(32), 3, "input");

    Func result = blur_cols_inductive(Func(A), row, 0.1f);

    Buffer<float> a_buf(acc, row, ch);
    fill_buffer_a_f32(a_buf, row, acc, ch);
    A.set(a_buf);

    Buffer<float> out(row, acc, ch);
    auto time = Tools::benchmark(20, 20, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

bool sum_normal(){
    const int length = 1<<20;
    ImageParam A(Int(32), 1, "input");

    Func result = sum_scan_normal(Func(A), length);

    Buffer<int> a_buf(length + 8);
    for (int i = 0; i < length + 8; ++i) {
        a_buf(i) = i + 1;
    }
    A.set(a_buf);

    Buffer<int> out(length);
    auto time = Tools::benchmark(20, 20, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

bool sum_inductive(){
    const int length = 1<<20;
    ImageParam A(Int(32), 1, "input");

    Func result = sum_scan_bad(Func(A), length);

    Buffer<int> a_buf(length + 8);
    for (int i = 0; i < length + 8; ++i) {
        a_buf(i) = i + 1;
    }
    A.set(a_buf);

    Buffer<int> out(length);
    auto time = Tools::benchmark(20, 20, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

int main(int argc, char **argv) {
    Target target = get_jit_target_from_environment();
    printf("Running normal IIR...\n");
    sum_normal();
    printf("Running inductive IIR...\n");
    sum_inductive();
    return 0;
}