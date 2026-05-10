#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"

#include "halide_image_io.h"

#include <cmath>
#include <cstdint>
#include <iostream>

#include <stdexcept>

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

void fill_buffer_2d(Buffer<float> &buf, int row, int acc) {
    for (int iy = 0; iy < row; ++iy) {
        for (int ix = 0; ix < acc; ++ix) {
            // value between 0 and 100
            float val = float(((float)rand() / (float)(RAND_MAX)) * 100.f);
            buf(ix, iy) = val;
        }
    }
    
}

void fill_buffer_2d_int(Buffer<int32_t> &buf, int row, int acc) {
    for (int iy = 0; iy < row; ++iy) {
        for (int ix = 0; ix < acc; ++ix) {
            // value between 0 and 100
            int32_t val = static_cast<int32_t>(((float)rand() / (float)(RAND_MAX)) * 100.f);
            buf(ix, iy) = val;
        }
    }
    
}

void fill_buffer_2d_int8(Buffer<uint8_t> &buf, int row, int acc) {
    for (int iy = 0; iy < row; ++iy) {
        for (int ix = 0; ix < acc; ++ix) {
            // value between 0 and 100
            int32_t val = static_cast<int32_t>(((float)rand() / (float)(RAND_MAX)) * 100.f);
            buf(ix, iy) = static_cast<uint8_t>(val);
        }
    }
    
}

void fill_buffer_1d_int(Buffer<int32_t> &buf, int row) {
    for (int iy = 0; iy < row; ++iy) {
        // value between 0 and 100
        int32_t val = static_cast<int32_t>(((float)rand() / (float)(RAND_MAX)) * 100.f);
        buf(iy) = val;
    }
    
}


Func sum_scan_inductive(Func input, Expr length) {
    Func f4 = Func(Int(32), "f4");
    Func f1("f1"), f2("f2"), f3("f3"), f5("f5"), in("in");
    Var x("x"), xo("xo"), xi("xi");
    // compute prefix sum in f4 in stages with SIMD
    in(x) = x + 1;
    f1(x) = input(x) + input(x + 3);
    f2(x) = f1(x) + f1(x + 2);
    f3(x) = f2(x) + f2(x + 2);
    f4(x) = select(x < 8, f3(x), likely(f3(x) + f4(x-8)));
    f5(x) = f4(x) / 4;
    f5.bound(x, 0, length).split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    f4.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi).store_root().compute_at(f5, xo).fold_storage(x, 16);
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
    f1(x) = input(x) + input(x + 3);
    f2(x) = f1(x) + f1(x + 2);
    f3(x) = f2(x) + f2(x + 2);
    f4(x) = select(x < 8, 0, f3(x));
    f5(x) = f4(x) / 4;
    f5.bound(x, 0, length).split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    //f4.bound(x, -8, 1024);
    //f3.bound(x, 0, 1025);
    //f2.bound(x, 0, 1026);
    //f1.bound(x, 0, 1028);
    
    //f5.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    f4.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi).compute_at(f5, xo).store_in(MemoryType::Register);
    for (Func f :  {f1, f2, f3}) {
        f.compute_at(f5, xo).vectorize(x).store_in(MemoryType::Register);//.fold_storage(x, 24);
    }
    //f3.fold_storage(x, 8);
    //f4.fold_storage(x, 16);

    return f5;
}


Func sum_scan_simple_inductive(Func input, Expr length) {
    Func f4 = Func(Int(32), "f4");
    Func f5("f5");
    Var x("x"), xo("xo"), xi("xi");
    RDom r(1, length / 8 - 1);
    RDom ri(0, 8);
    // compute prefix sum in f4 in stages with SIMD


    f4(x) = select(x < 1, input(x), likely(input(x) + f4(x - 1)));
    // divide prefix sum by 4 to get the final result
    f5(x) = f4(x) / 4;
    f5.bound(x, 0, length);//.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    f4.store_root().compute_at(f5, x).fold_storage(x, 2);
    return f5;
}

Func sum_scan_simple_rdom(Func input, Expr length) {
    Func f4 = Func(Int(32), "f4");
    Func f5("f5");
    Var x("x"), xo("xo"), xi("xi");
    RDom r(0, length);
    RDom ri(0, 8);
    // compute prefix sum in f4 in stages with SIMD


    f4(x) = undef<int>();
    f4(0) = input(0);
    f4(r) = input(r) + f4(r-1);
    // divide prefix sum by 4 to get the final result
    f5(x) = f4(x) / 4;
    f5.bound(x, 0, length).split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    f4.compute_root();
    return f5;
}


Func sum_scan_normal(Func input, Expr length) {
    Func f4 = Func(Int(32), "f4");
    Func f1("f1"), f2("f2"), f3("f3"), f5("f5"), in("in");
    Var x("x"), xo("xo"), xi("xi");
    // compute prefix sum in f4 in stages with SIMD
    in(x) = x + 1;
    f1(x) = input(x) + input(x + 4);
    f2(x) = f1(x) + f1(x + 2);
    f3(x) = f2(x) + f2(x + 1);
    f4(x) = select(x < 8, f3(x), likely(f3(x) + f4(x-8)));
    // divide prefix sum by 4 to get the final result
    f5(x) = f4(x) / 4;
    f5.bound(x, 0, length).split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    f4.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi).compute_root();
    for (Func f :  {f1, f2, f3}) {
        f.compute_at(f4, xo).vectorize(x).store_in(MemoryType::Register).fold_storage(x, 16);
    }
    return f5;
}

Func sum_scan_normal_rvar(Func input, Expr length) {
    Func f4 = Func(Int(32), "f4");
    Func f1("f1"), f2("f2"), f3("f3"), f5("f5"), in("in");
    Var x("x"), xo("xo"), xi("xi");
    RDom r(8, length);
    RDom r0(1, 7);
    RVar ri("ri"), ro("ro");
    // compute prefix sum in f4 in stages with SIMD
    in(x) = x + 1;
    f1(x) = input(x) + input(x + 4);
    f2(x) = f1(x) + f1(x + 2);
    f3(x) = f2(x) + f2(x + 1);
    f4(x) = undef<int>();
    f4(0)=0;
    f4(r0) = input(r0)+f4(r0-1);
    f4(r) = f3(r) + f4(r-8);
    //f4(x) = select(x < 8, f3(x)-f3(x), likely(f3(x) + f4(x-8)));
    // divide prefix sum by 4 to get the final result
    f5(x) = f4(x) / 4;
    f5.bound(x, 0, length).split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    f4.compute_root();
    f4.update(1).split(r,ro,ri,8).allow_race_conditions().vectorize(ri,8);//.compute_root();
    for (Func f :  {f1, f2, f3}) {
        f.compute_at(f4, ro).vectorize(x).store_in(MemoryType::Register).fold_storage(x, 16);
    }
    return f5;
}

Func blur_cols_inductive(Func input, Expr height, Expr alpha) {
    Var x, y, c;

    const int vec = 8;
    

    // Pure definition: do nothing.
    Func blur = Func(Float(32), "blur");
    blur(x, y, c) = select(y <= 0, input(x, y, c), likely((1 - alpha) * blur(x, y - 1, c) + alpha * input(x, y, c)));
    
    Func blur2 = Func(Float(32), "blur2");
    blur2(x, y, c) = select(y <= 0, blur(x, height - 1, c), likely((1 - alpha) * blur2(x, y - 1, c) + alpha * blur(x, height - y - 1, c)));

    Func transpose("transpose");
    transpose(x, y, c) = undef<float>();

    RDom rx(0, height);
    
    transpose(height-rx-1, y, c) = blur2(y,rx, c);

    Var xo, yo,xi,yi, t;
    transpose.bound(y, 0, height);
    transpose.bound(x, 0, height);
    RVar rxi, rxo;
    transpose.compute_root().update(0)
        .tile(rx, y, rxo, yo, rxi, yi, 8, vec * 4)
        //.parallel(yo)
        //.parallel(c)
        .vectorize(rxi)
        ;
    blur.compute_at(transpose, yo).reorder(x, y).vectorize(x);
    blur2.compute_at(transpose, rxo).store_at(transpose, yo).reorder(x, y).vectorize(x);//.fold_storage(y, 16);
    //
    return transpose;
}

Func blur_cols_reg(Func input, Expr height, Expr alpha) {
    Var x, y, c;

    const int vec = 8;
    

    // Pure definition: do nothing.
    Func blur = Func(Float(32), "blur");
    //blur(x, y, c) = select(y <= 0, input(x, y, c), likely((1 - alpha) * blur(x, y - 1, c) + alpha * input(x, y, c)));
    

    // Pure definition: do nothing.
    blur(x, y, c) = undef<float>();
    // Update 0: set the top row of the result to the input.
    blur(x, 0, c) = input(x, 0, c);
    // Update 1: run the IIR filter down the columns.
    RDom ry(1, height - 1);
    blur(x, ry, c) =
        (1 - alpha) * blur(x, ry - 1, c) + alpha * input(x, ry, c);

    Func blur2 = Func(Float(32), "blur2");
    blur2(x, y, c) = blur(x, y, c)/4;

    Func transpose("transpose");
    //transpose(x, y, c) = undef<float>();

    RDom rx(0, height);
    
    /*transpose(rx, y, c) = blur2(y,height-rx-1, c);

    Var xo, yo,xi,yi, t;
    transpose.bound(y, 0, height);
    transpose.bound(x, 0, height);
    RVar rxi, rxo;
    transpose.compute_root().update(0)
        .tile(rx, y, rxo, yo, rxi, yi, 8, vec * 4)
        //.parallel(yo)
        //.parallel(c)
        .vectorize(rxi)
        ;*/

    transpose(x, y, c) = blur2(y, x, c);

    // Schedule

    // CPU schedule.
    // 8.2ms on an Intel i9-9960X using 16 threads
    // Split the transpose into tiles of rows. Parallelize over channels
    // and strips (Halide supports nested parallelism).
    Var xo, yo, t;
    transpose.compute_root()
        .tile(x, y, xo, yo, x, y, vec, vec * 4)
        .vectorize(x)
        //.parallel(yo)
        //.parallel(c)
        ;
    //blur.compute_at(transpose, yo).reorder(x, y).vectorize(x);
    
    blur.compute_at(transpose, yo);

    blur.reorder(x,y).vectorize(x);

    // Vectorize computations within the strips.
    blur.update(0)
        .unscheduled();
    blur.update(1)
        .reorder(x, ry)
        .vectorize(x)
        ;
    blur2.compute_at(transpose, xo).reorder(x, y).vectorize(x).store_in(MemoryType::Register);
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
        .vectorize(x)
        //.parallel(yo)
        //.parallel(c)
        ;

    // Run the filter on each row of tiles (which corresponds to a strip of
    // columns in the input).
    blur.compute_at(transpose, yo);

    blur.reorder(x,y).vectorize(x);

    // Vectorize computations within the strips.
    blur.update(0)
        .unscheduled();
    blur.update(1)
        .reorder(x, ry)
        .vectorize(x)
        ;
    blur.update(2)
        .reorder(x, ry)
        .vectorize(x)
        ;
        
        
    return transpose;
}

Func box_blur(Func input, Expr height, const int dist){
    Var x("x"), y("y"), c;
    Func blur_x(Float(32), "blur_x");
    RDom rx(0, dist);
    Func zero_blur("zero_blur");
    zero_blur(y,c) = sum(input(rx, y, c));

    blur_x(x,y,c) = select(x<=0, zero_blur(y,c), likely(blur_x(x-1, y, c) + input(x + dist -1, y, c) - input(max(x-1,0), y, c))); /// should be x-1
    Func blur_y("blur_y");
    RDom ry(0, height-dist, "ry");
    blur_y(x,y,c) = undef<float>();
    Func f1("f1");
    f1(x,c) = sum(blur_x(x,rx,c));
    blur_y(x,ry,c) = select(ry == 0, f1(x,c), likely(blur_y(x,ry-1,c) + blur_x(x, ry + dist -1, c) - blur_x(x, max(ry-1,0) , c)));
    //should be ry-1

    Func out("out");
    out(x,y,c) = blur_y(x,y,c)/ (dist * dist);

    RVar ryi("ryi"),ryo("ryo");
    zero_blur.compute_root().vectorize(y,8);
    blur_y.reorder_storage(y,x,c);
    blur_x.reorder_storage(y,x,c);
    blur_y.compute_root().update(0).split(ry,ryo,ryi,8).reorder(ryi,ryo,x,c);
    blur_x.compute_at(blur_y, ryo).vectorize(y,8).store_at(blur_y, c).fold_storage(x,2);
    return out;
}

Func box_blur_sim(Func input, Expr height){
    Var x("x"), y("y"), c;
    Func blur_x(Float(32), "blur_x");
    blur_x(x,y,c)=input(x,y)*c*c;
    Func blur_y("blur_y");
    blur_y(x,y) = 0.0f;
    RDom ry(0, 16);
    blur_y(x,y) = blur_y(x,y) + blur_x(x,y,ry);

    blur_x.compute_root().vectorize(x,8);
    blur_y.vectorize(x,8);
    blur_y.update(0).reorder(x,y,ry).vectorize(x,8);

    return blur_y;
}

Func box_blur_sim2(Func input, Expr height){
    Var x("x"), y("y"), c;
    Func blur_x(Float(32), "blur_x");
    blur_x(x,y,c)=input(x,y)*c*c;
    Func blur_y("blur_y");
    blur_y(x,y) = 0.0f;
    RDom ry(0, 16);
    blur_y(x,y) = blur_y(x,y) + blur_x(x,y,ry);

    RVar ryi("ryi"),ryo("ryo");

    blur_y.vectorize(x,8);
    blur_y.update(0).reorder(x,y,ry).vectorize(x,8).split(ry,ryo,ryi,2);
    blur_x.compute_at(blur_y, ryo).vectorize(x,8);

    return blur_y;
}

Func box_blur_sim3(Func input, Expr height){
    Var x("x"), y("y"), c;
    Func blur_x(Float(32), "blur_x");
    blur_x(x,y,c)=input(x,y)*c*c;
    Func blur_y("blur_y");
    blur_y(x,y) = 0.0f;
    RDom ry(0, 16);
    blur_y(x,y) = blur_y(x,y) + blur_x(x,y,ry);

    RVar ryi("ryi"),ryo("ryo");
    Var yo("yo"), yi("yi");

    blur_y.vectorize(x,8);
    blur_y.update(0).split(y,yo,yi, 32).reorder(x,yi,ry,yo).vectorize(x,8);
    blur_x.compute_at(blur_y, yo).vectorize(x,8);

    return blur_y;
}

Func box_blur_sim4(Func input, Expr height){
    Var x("x"), y("y"), c;
    Func blur_x(Float(32), "blur_x");
    blur_x(x,y,c)=input(x,y)*c*c+input(x,y)*input(x,y);
    Func blur_y("blur_y");
    blur_y(x,y) = 0.0f;
    RDom ry(0, 16);
    blur_y(x,y) = blur_y(x,y) + blur_x(x,y,ry);

    RVar ryi("ryi"),ryo("ryo");
    Var yo("yo"), yi("yi");

    blur_y.vectorize(x,8);
    blur_y.update(0).split(y,yo,yi, 64).reorder(x,yi,ry,yo).vectorize(x,8);
    blur_x.compute_at(blur_y, ry).vectorize(x,8);

    return blur_y;
}

bool box_blur_s1(){
    const int row = 1024;
    const int acc = 64;
    const int ch = 3;
    const int dist = 8;

    ImageParam A(Float(32), 2, "input");

    Func result = box_blur_sim3(Func(A), 64);

    Buffer<float> a_buf(acc, row);
    fill_buffer_2d(a_buf, row, acc);
    A.set(a_buf);

    Buffer<float> out(acc, row);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

bool box_blur_s2(){
    const int row = 1024;
    const int acc = 64;
    const int ch = 3;
    const int dist = 8;

    ImageParam A(Float(32), 2, "input");

    Func result = box_blur_sim4(Func(A), 64);

    Buffer<float> a_buf(acc, row);
    fill_buffer_2d(a_buf, row, acc);
    A.set(a_buf);

    Buffer<float> out(acc, row);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

bool box_blur_normal(){
    const int row = 2048;
    const int acc = 2048;
    const int ch = 3;
    const int dist = 8;

    ImageParam A(Float(32), 3, "input");

    Func result = box_blur(Func(A), row, dist);

    Buffer<float> a_buf(acc, row, ch);
    fill_buffer_a_f32(a_buf, row, acc, ch);
    A.set(a_buf);

    Buffer<float> out(row-16, acc-16, ch);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

bool iir_normal() {
    const int row = 2016;
    const int acc = 2016;
    const int ch = 1;

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
    const int ch = 1;

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
    const int length = 1<<26;
    ImageParam A(Int(32), 1, "input");

    Func result = sum_scan_normal(Func(A), length);

    Buffer<int> a_buf(length + 16);
    for (int i = 0; i < length + 16; ++i) {
        a_buf(i) = i + 1;
    }
    A.set(a_buf);

    Buffer<int> out(length);
    auto time = Tools::benchmark(10, 10, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

bool sum_inductive(){
    const int length = 1<<26;
    ImageParam A(Int(32), 1, "input");

    Func result = sum_scan_inductive(Func(A), length);

    Buffer<int> a_buf(length + 16);
    for (int i = 0; i < length + 16; ++i) {
        a_buf(i) = i + 1;
    }
    A.set(a_buf);

    Buffer<int> out(length);
    auto time = Tools::benchmark(10, 10, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}




Func diff_blur_good(ImageParam input0, int height, int winsize, int depth, int tilesize){
     Func output("output");
    Type int32 = Int(32);
    const int native_lanes = 8;
    //const Expr depth = input.channels();
    Func f("f");
    Func g("g");
    Var x("x");
    Var y("y");
    Var d("d");
    Var di("di");
    Var xi("xi"), yi("yi"),xo("xo"),yo("yo");

    Func diff("diff");


    Func b0("b0"), b1("b1");

        b0(y,x)=BoundaryConditions::constant_exterior(input0, 0)(y,x);

    b0.compute_root().vectorize(y,native_lanes);

    // cast to int not uint
    diff(di,yi,x,yo)=Halide::cast<int32_t>(abs(b0(yi+yo*tilesize,x)-b0(yi+yo*tilesize+di,x)));
    Func vsum(int32, "vsum");
    Func zero_blur("zero_blur");

    RDom rx0(0, winsize);

    zero_blur(di, yi, yo) = sum(diff(di, yi, rx0, yo));
    vsum(di,yi,x,yo) = select(x<=0, zero_blur(di, yi, yo), likely(vsum(di, yi, max(x-1,0), yo) + diff(di, yi, x+winsize-1, yo) - diff(di, yi, max(x-1, 0), yo))); /// should be x-1
    
    Func blur_y("blur_y");
    RDom ry(0, tilesize, "ry");
    blur_y(di,yi,x,yo) = undef<int32_t>();
    Func f1("f1");
    f1(di, x, yo) = sum(vsum(di,rx0, x, yo));
    blur_y(di,ry,x,yo) = select(ry == 0, f1(di, x, yo), likely(blur_y(di, max(ry - 1, 0), x, yo) + vsum(di, ry + winsize - 1, x, yo) -vsum(di, max(ry - 1, 0), x, yo)));
    //should be ry-1

    Func preout("preout");
    //preout(di,y,x)=blur_y(di,y%tilesize,x%tilesize,y/tilesize,x/tilesize);

    RDom rd(0,depth,"rd");
    //Func argminfunc;
    preout(yi,x,yo)=Tuple(10000, 0);

    preout(yi,x,yo)=select(preout(yi,x,yo)[0]<blur_y(rd,yi,x,yo), preout(yi,x,yo), Tuple(blur_y(rd,yi,x,yo), rd));

    Func output2("output2");
    output2(y,x)=preout(y%tilesize,x,y/tilesize)[1];
    
    
    preout.compute_root();
    RVar ri("ri"), ro("ro");
    preout.update().split(rd, ro, ri, native_lanes);
    Var dii("dii");
    // print out output.update dimensions:
    Func intermediate = preout.update().rfactor({{ri, dii}});

    intermediate.compute_at(preout, x);

    intermediate.reorder(dii,yi,x,yo).vectorize(dii,native_lanes).reorder_storage(dii,yi,x,yo).update().reorder(dii, yi, x, ro).vectorize(dii, native_lanes);


    //argminfunc2.compute_root().update().reorder(yi,rd, x, yo);
    //argminfunc.compute_at(argminfunc2,x).update().reorder(di,rdi,yi).vectorize(di,native_lanes);
    blur_y.compute_at(preout,x).update().reorder(di,ry).vectorize(di,native_lanes);
    vsum.compute_at(preout,x).store_at(preout, yo).vectorize(di,native_lanes).fold_storage(x,2);
    f1.compute_at(preout,x).vectorize(di, 8);
    zero_blur.compute_at(preout, yo).vectorize(di, 8);

    output2.vectorize(y,native_lanes);

    return output2;

}



Func diff_blur_bad(ImageParam input0, int height, int winsize, int depth, int tilesize){
    Func output("output");
    Type int32 = Int(32);
    Type uint32 = UInt(32);
    const int native_lanes = 8;
    //const Expr depth = input.channels();
    Func f("f");
    Func g("g");
    Var x("x");
    Var y("y");
    Var d("d");
    Var di("di");
    Var xi("xi"), yi("yi"),xo("xo"),yo("yo");

    Func diff("diff");

    Func b0("b0"), b1("b1");

        b0(y,x)=BoundaryConditions::constant_exterior(input0, 0)(y,x);

    b0.compute_root().vectorize(y,native_lanes);

    // cast to int not uint
    diff(di,yi,xi,yo,xo)=Halide::cast<int32_t>(abs(b0(yi+yo*tilesize,xi+xo*tilesize)-b0(yi+yo*tilesize+di,xi+xo*tilesize)));

    RDom rx(1, tilesize-1, "rx");

    RDom rk(0,winsize,"rk");
    Func vsum("vsum");
    vsum(di,yi,xi,yo,xo) = undef<int>();
    vsum(di,yi,0,yo,xo) = sum(diff(di,yi,rk,yo,xo));
    vsum(di,yi,rx,yo,xo)=vsum(di,yi,rx-1,yo,xo)+diff(di,yi,rx+winsize-1,yo,xo)-diff(di,yi,rx-1,yo,xo);

    g(di,yi,xi,yo,xo) = undef<int>();
    g(di,0,xi,yo,xo) = sum(vsum(di,rk,xi,yo,xo));
    g(di,rx,xi,yo,xo) = g(di,rx-1,xi,yo,xo)+ vsum(di,rx+tilesize-1,xi,yo,xo)-vsum(di,rx-1,xi,yo,xo);

    Func preout("preout");





    preout(di,y,x)=g(di,y%tilesize,x%tilesize,y/tilesize,x/tilesize);

    //g.compute_root();

    

    

    RDom rd(0,depth,"rd");
    Func argminfunc;
    output(y,x)=Tuple(10000, 0);

    output(y,x)=select(output(y,x)[0]<preout(rd,y,x), output(y,x), Tuple(preout(rd,y,x), rd));

    Func output2("output2");
    output2(y,x)=output(y,x)[1];
    
    
    output.compute_root().vectorize(y,native_lanes);
    RVar ri("ri"), ro("ro");
    output.update().split(rd, ro, ri, native_lanes);
    Var dii("dii");
    // print out output.update dimensions:
    Func intermediate = output.update().rfactor({{ri, dii}});

    intermediate.reorder_storage(dii,y,x).update().reorder(dii, y, x, ro).vectorize(dii, native_lanes);
    intermediate.reorder(dii,y,x);

    //output(y,x)=sum(preout(rd,y,x));


    vsum.compute_at(intermediate, ro).reorder(di,yi,xi,yo,xo).vectorize(di,native_lanes).update().reorder(di,yi,yo,xo).vectorize(di,native_lanes);
    vsum.update(1).reorder(di,yi,rx,yo,xo).vectorize(di,native_lanes);

    g.reorder(yo,xi,xo).vectorize(yi,native_lanes).update().vectorize(di,native_lanes);
    g.update().reorder(yo,xi,xo);
    g.update(1).reorder(rx,yo,xi,xo).vectorize(di,native_lanes);

    output.tile(y,x,yo,xo,yi,xi,tilesize,tilesize);
    output.update().tile(y,x,yo,xo,yi,xi,tilesize,tilesize).reorder(yi,xi,ri,yo,xo);
    intermediate.compute_at(output, yo).vectorize(dii, native_lanes);

    g.compute_at(intermediate,x);

    output2.vectorize(y,native_lanes);

    return output2;
}


Func diff_blur_bad2(ImageParam input0, ImageParam input1, int height, int winsize, int depth, int tilesize){
    Func output("output");
    Type int32 = Int(32);
    Type uint32 = UInt(32);
    const int native_lanes = 8;
    //const Expr depth = input.channels();
    Func f("f");
    Func g("g");
    Var x("x");
    Var y("y");
    Var d("d");
    Var di("di");
    Var xi("xi"), yi("yi"),xo("xo"),yo("yo");

    Func diff("diff");

    Func b0("b0"), b1("b1");

    b0(y,x)=BoundaryConditions::constant_exterior(input0, 0)(y,x);

    b0.compute_root().vectorize(y,native_lanes);

    b1(y,x)=BoundaryConditions::constant_exterior(input1, 0)(y,x);

    b1.compute_root().vectorize(y,native_lanes);

    // cast to int not uint
    diff(di,yi,xi,yo,xo)=Halide::cast<int32_t>(abs(b0(yi+yo*tilesize,xi+xo*tilesize)-b1(yi+yo*tilesize+di,xi+xo*tilesize)));

    RDom rx(1, tilesize-1, "rx");

    RDom rk(0,winsize,"rk");
    Func vsum("vsum");
    vsum(di,yi,xi,yo,xo) = undef<int>();
    vsum(di,yi,0,yo,xo) = sum(diff(di,yi,rk,yo,xo));
    vsum(di,yi,rx,yo,xo)=vsum(di,yi,rx-1,yo,xo)+diff(di,yi,rx+winsize-1,yo,xo)-diff(di,yi,rx-1,yo,xo);

    g(di,yi,xi,yo,xo) = undef<int>();
    g(di,0,xi,yo,xo) = sum(vsum(di,rk,xi,yo,xo));
    g(di,rx,xi,yo,xo) = g(di,rx-1,xi,yo,xo)+ vsum(di,rx+tilesize-1,xi,yo,xo)-vsum(di,rx-1,xi,yo,xo);

    Func preout("preout");

    preout(yi,xi,yo,xo)=Tuple(10000, 0);

    RDom rd(0,depth,"rd");

    preout(yi,xi,yo,xo)=select(preout(yi,xi,yo,xo)[0]<g(rd,yi,xi,yo,xo), preout(yi,xi,yo,xo), Tuple(g(rd,yi,xi,yo,xo), rd));

    Func output0("output0");
    output0(yi,xi,yo,xo)=preout(yi,xi,yo,xo)[1];

    Func output2("output2");
    output2(y,x)=output0(y%tilesize,x%tilesize,y/tilesize,x/tilesize);
    
    
    output0.compute_root();
    RVar ri("ri"), ro("ro");
    preout.update().split(rd, ro, ri, native_lanes);
    Var dii("dii");
    // print out output.update dimensions:
    Func intermediate = preout.update().rfactor({{ri, dii}});

    intermediate.compute_at(output0, yo).vectorize(dii, native_lanes);

    intermediate.reorder(dii,yi,xi,yo,xo).reorder_storage(dii,yi,xi,yo,xo).update().reorder(dii, yi, xi, ro).vectorize(dii, native_lanes);


    vsum.compute_at(output0, yo).reorder(di,yi,xi,yo,xo).vectorize(di,native_lanes).update().reorder(di,yi,yo,xo).vectorize(di,native_lanes);
    vsum.update(1).reorder(di,yi,rx,yo,xo).vectorize(di,native_lanes);

    g.vectorize(yi,native_lanes).update().vectorize(di,native_lanes);
    g.update().reorder(yo,xi,xo);
    g.update(1).reorder(rx,yo,xi,xo).vectorize(di,native_lanes);

    g.compute_at(output0, yo);

    preout.update().reorder(yi,xi,ri,yo,xo);


    return output2;
}


bool diff_blur_s1(){
    const int row = 1024;
    const int acc = 1024;
    const int depth = 64; 
    const int winsize = 10;
    const int tilesize = 64;

    ImageParam A(Int(32), 2, "input0");
    ImageParam B(Int(32), 2, "input1");

    Func result = diff_blur_bad2(A, B, 1024, winsize, depth, tilesize);

    Buffer<int32_t> a_buf(acc, row);
    fill_buffer_2d_int(a_buf, row, acc);
    A.set(a_buf);

    Buffer<int32_t> b_buf(acc, row);
    fill_buffer_2d_int(b_buf, row, acc);
    B.set(b_buf);
    Buffer<int32_t> out(acc, row);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

bool diff_blur_s2(){
    const int row = 1024;
    const int acc = 1024;
    const int depth = 64; 
    const int winsize = 10;
    const int tilesize = 64;

    ImageParam A(Int(32), 2, "input");

    Func result = diff_blur_good(A, 1024, winsize, depth, tilesize);

    Buffer<int32_t> a_buf(acc, row);
    fill_buffer_2d_int(a_buf, row, acc);
    A.set(a_buf);

    Buffer<int32_t> out(acc, row);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}


Func diff_blur_1d_inner_good(ImageParam input0, ImageParam input1, int height, int winsize, int depth, int tilesize){
     Func output("output");
    Type int32 = Int(32);
    const int native_lanes = 8;
    //const Expr depth = input.channels();
    Func f("f");
    Func g("g");
    Var y("y");
    Var d("d");
    Var di("di");
    Var yi("yi"),yo("yo");

    Func diff("diff");


    Func b0("b0"), b1("b1");

    b0(y)=BoundaryConditions::constant_exterior(input0, 0)(y);

    b0.compute_root().vectorize(y,native_lanes);

    b1(y)=BoundaryConditions::constant_exterior(input1, 0)(y);

    b1.compute_root().vectorize(y,native_lanes);

    // cast to int not uint
    diff(di,y)=Halide::cast<int32_t>(abs(b1(y+di)-b0(y)));

    RDom rx0(0, winsize);

    
    Func blur_y(Int(32), "blur_y");
    //RDom ry(0, tilesize, "ry");
    Func f1("f1");
    f1(di, y) = sum(diff(di,rx0));
    blur_y(di,y) = select(y <= 0, f1(di, y), likely(blur_y(di, max(y - 1, 0)) + diff(di, y + winsize - 1) -diff(di, max(y - 1, 0))));
    //should be ry-1

    Func preout("preout");
    //preout(di,y,x)=blur_y(di,y%tilesize,x%tilesize,y/tilesize,x/tilesize);

    RDom rd(0,depth,"rd");
    //Func argminfunc;
    preout(y)=Tuple(10000, 0);

    preout(y)=select(preout(y)[0]<blur_y(rd,y), preout(y), Tuple(blur_y(rd,y), rd));

    Func output2("output2");
    output2(y)=preout(y)[1];
    
    
    preout.compute_root();
    RVar ri("ri"), ro("ro");
    preout.update().split(rd, ro, ri, native_lanes);
    Var dii("dii");
    // print out output.update dimensions:
    Func intermediate = preout.update().rfactor({{ri, dii}});

    intermediate.compute_at(preout, y);

    intermediate.reorder(dii,y).vectorize(dii,native_lanes).reorder_storage(dii,y).update().reorder(dii, y, ro).vectorize(dii, native_lanes);


    //argminfunc2.compute_root().update().reorder(yi,rd, x, yo);
    //argminfunc.compute_at(argminfunc2,x).update().reorder(di,rdi,yi).vectorize(di,native_lanes);
    blur_y.compute_at(preout,y).store_root().reorder(di,y).vectorize(di,native_lanes).fold_storage(y,2);
    //vsum.compute_at(preout,x).store_at(preout, yo).vectorize(di,native_lanes).fold_storage(x,2);
    f1.compute_root().vectorize(di, 8);

    return output2;

}

Func diff_blur_1d_inner_good_full(ImageParam input0, ImageParam input1, int height, int winsize, int depth, int tilesize){
     Func output("output");
    Type int32 = Int(32);
    const int native_lanes = 8;
    //const Expr depth = input.channels();
    Func f("f");
    Func g("g");
    Var y("y");
    Var d("d");
    Var di("di");
    Var yi("yi"),yo("yo");

    Func diff("diff");


    Func b0("b0"), b1("b1");

    b0(y)=BoundaryConditions::constant_exterior(input0, 0)(y);

    b0.compute_root().vectorize(y,native_lanes);

    b1(y)=BoundaryConditions::constant_exterior(input1, 0)(y);

    b1.compute_root().vectorize(y,native_lanes);

    // cast to int not uint
    diff(di,y)=Halide::cast<int32_t>(abs(b1(y+di)-b0(y)));

    RDom rx0(0, winsize);

    
    Func blur_y(Int(32), "blur_y");
    //RDom ry(0, tilesize, "ry");
    Func f1("f1");
    f1(di) = sum(diff(di,rx0));
    blur_y(di,y) = select(y <= 0, f1(di), likely(blur_y(di, max(y - 1, 0)) + diff(di, y + winsize - 1) -diff(di, max(y - 1, 0))));
    //should be ry-1

    const int filterzero = 10;
    Func text("text"), textzero("textzero"), textsum("textsum");
    text(y) = cast<int>(abs(b0(y)-filterzero));
    textzero() = sum(text(rx0));
    textsum(y) = select(y<=0, textzero(), textsum(max(y-1,0))+ text(y+winsize-1)-text(max(y-1, 0)));



    Func preout("preout");

    preout(y)=Tuple(10000, 0);

    RDom rd(0,depth,"rd");

    const int threshold = 50;
    const int mindisp = 0;

    preout(y)=select(preout(y)[0]<blur_y(rd,y), preout(y), Tuple(blur_y(rd,y), rd));

    Func p_clamped("p_clamped");
    p_clamped(y) = unsafe_promise_clamped(preout(y)[1], 0, depth);

    Func subpout("subpout");
    Expr p = blur_y(depth-1-cast(int32, abs(depth-2-p_clamped(y))), y);
    Expr n = blur_y(cast(int32, abs(p_clamped(y)-1)), y);
    Expr d1 = p + n - 2*preout(y)[0] + abs(p - n);
    subpout(y) = (depth-preout(y)[1]-1+mindisp)*256 + (select(d1==0, 0, (p-n)*256/d1)+15)/16;

    blur_y.bound(di,0,depth);
    

    Func output0("output0");
    output0(y)=select(textsum(y)<threshold, -1, subpout(y));



    //Func output2("output2");
    //output2(y)=output0(y%tilesize,y/tilesize);



    
    
    output0.compute_root();
    RVar ri("ri"), ro("ro");
    preout.update().split(rd, ro, ri, native_lanes);
    Var dii("dii");
    Func intermediate = preout.update().rfactor({{ri, dii}});

    intermediate.compute_at(output0, y);
    intermediate.reorder(dii,y).vectorize(dii,native_lanes).reorder_storage(dii,y).update().reorder(dii, y, ro).vectorize(dii, native_lanes);
    blur_y.compute_at(output0,y).store_root().reorder(di,y).vectorize(di,native_lanes).fold_storage(y,2);
    textsum.compute_at(output0, y).store_root().fold_storage(y,2);
    subpout.compute_at(output0, y);//.vectorize(yi, native_lanes);

    preout.update().reorder(y,ri);

    /*Func preout("preout");
    //preout(di,y,x)=blur_y(di,y%tilesize,x%tilesize,y/tilesize,x/tilesize);

    RDom rd(0,depth,"rd");
    //Func argminfunc;
    preout(y)=Tuple(10000, 0);

    preout(y)=select(preout(y)[0]<blur_y(rd,y), preout(y), Tuple(blur_y(rd,y), rd));

    Func output2("output2");
    output2(y)=preout(y)[1];
    
    
    preout.compute_root();
    RVar ri("ri"), ro("ro");
    preout.update().split(rd, ro, ri, native_lanes);
    Var dii("dii");
    // print out output.update dimensions:
    Func intermediate = preout.update().rfactor({{ri, dii}});

    intermediate.compute_at(preout, y);

    intermediate.reorder(dii,y).vectorize(dii,native_lanes).reorder_storage(dii,y).update().reorder(dii, y, ro).vectorize(dii, native_lanes);


    //argminfunc2.compute_root().update().reorder(yi,rd, x, yo);
    //argminfunc.compute_at(argminfunc2,x).update().reorder(di,rdi,yi).vectorize(di,native_lanes);
    blur_y.compute_at(preout,y).store_root().reorder(di,y).vectorize(di,native_lanes).fold_storage(y,2);
    //vsum.compute_at(preout,x).store_at(preout, yo).vectorize(di,native_lanes).fold_storage(x,2);
    f1.compute_root().vectorize(di, 8);*/

    return output0;

}


bool diff_blur_1d_good(){
    const int row = 1024*1024;
    const int acc = 1024;
    const int depth = 64; 
    const int winsize = 10;
    const int tilesize = 256;

    ImageParam A(Int(32), 1, "input0");
    ImageParam B(Int(32), 1, "input1");

    Func result = diff_blur_1d_inner_good_full(A, B, row, winsize, depth, tilesize);

    Buffer<int32_t> a_buf(row);
    fill_buffer_1d_int(a_buf, row);
    A.set(a_buf);

    Buffer<int32_t> b_buf(row);
    fill_buffer_1d_int(b_buf, row);
    B.set(b_buf);

    Buffer<int32_t> out(row);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}


Func diff_blur_1d_inner_bad(ImageParam input0, ImageParam input1, int height, int winsize, int depth, int tilesize){

    Func output("output");
    Type int32 = Int(32);
    const int native_lanes = 8;
    //const Expr depth = input.channels();
    Func f("f");
    Func g("g");
    Var y("y");
    Var d("d");
    Var di("di");
    Var yi("yi"),yo("yo");

    Func diff("diff");


    Func b0("b0"), b1("b1");

    b0(y)=BoundaryConditions::constant_exterior(input0, 0)(y);
    b0.compute_root().vectorize(y,native_lanes);
    b1(y)=BoundaryConditions::constant_exterior(input1, 0)(y);
    b1.compute_root().vectorize(y,native_lanes);

    // cast to int not uint
    diff(di,yi,yo)=Halide::cast<int32_t>(abs(b0(yi+yo*tilesize)-b0(yi+yo*tilesize+di)));

    RDom rx(1, tilesize-1, "rx");

    RDom rk(0,winsize,"rk");

    g(di,yi,yo) = undef<int>();
    g(di,0,yo) = sum(diff(di,rk,yo));
    g(di,rx,yo) = g(di,rx-1,yo)+ diff(di,rx+tilesize-1,yo)-diff(di,rx-1,yo);

    Func preout("preout");

    preout(yi,yo)=Tuple(10000, 0);

    RDom rd(0,depth,"rd");

    preout(yi,yo)=select(preout(yi,yo)[0]<g(rd,yi,yo), preout(yi,yo), Tuple(g(rd,yi,yo), rd));
    Func output2("output2");
    output2(y)=preout(y%tilesize,y/tilesize)[1];
    
    
    preout.compute_root();
    RVar ri("ri"), ro("ro");
    preout.update().split(rd, ro, ri, native_lanes);
    Var dii("dii");
    Func intermediate = preout.update().rfactor({{ri, dii}});

    intermediate.compute_at(preout, yo).vectorize(dii, native_lanes);

    intermediate.reorder(dii,yi,yo).reorder_storage(dii,yi,yo).update().reorder(dii, yi, ro).vectorize(dii, native_lanes);


    g.vectorize(di,native_lanes).update().vectorize(di,native_lanes);
    //g.update().reorder(yo);
    g.update(1).reorder(rx,yo).vectorize(di,native_lanes);

    g.compute_at(intermediate, ro);

    preout.update().reorder(yi,ri,yo);


    return output2;
}

Func diff_blur_1d_inner_bad_full(ImageParam input0, ImageParam input1, int height, int winsize, int depth, int tilesize){

    Func output("output");
    Type int32 = Int(32);
    const int native_lanes = 8;
    //const Expr depth = input.channels();
    Func f("f");
    Func g("g");
    Var y("y");
    Var d("d");
    Var di("di");
    Var yi("yi"),yo("yo");

    Func diff("diff");


    Func b0("b0"), b1("b1");

    b0(y)=BoundaryConditions::constant_exterior(input0, 0)(y);
    b0.compute_root().vectorize(y,native_lanes);
    b1(y)=BoundaryConditions::constant_exterior(input1, 0)(y);
    b1.compute_root().vectorize(y,native_lanes);

    // cast to int not uint
    diff(di,yi,yo)=Halide::cast<int32_t>(abs(b0(yi+yo*tilesize)-b0(yi+yo*tilesize+di)));

    RDom rx(1, tilesize-1, "rx");

    RDom rk(0,winsize,"rk");

    const int filterzero = 10;
    Func text("text"), textsum("textsum");
    text(yi,yo) = cast<int>(abs(b0(yi+yo*tilesize)-filterzero));
    textsum(yi,yo) = sum(text(rk,yo));
    textsum(rx,yo) = textsum(rx-1,yo)+ text(rx+winsize-1,yo)-text(rx-1,yo);


    g(di,yi,yo) = undef<int>();
    g(di,0,yo) = sum(diff(di,rk,yo));
    g(di,rx,yo) = g(di,rx-1,yo)+ diff(di,rx+tilesize-1,yo)-diff(di,rx-1,yo);

    Func preout("preout");

    preout(yi,yo)=Tuple(10000, 0);

    RDom rd(0,depth,"rd");

    const int threshold = 50;
    const int mindisp = 0;

    preout(yi,yo)=select(preout(yi,yo)[0]<g(rd,yi,yo), preout(yi,yo), Tuple(g(rd,yi,yo), rd));

    Func p_clamped("p_clamped");
    p_clamped(yi, yo) = unsafe_promise_clamped(preout(yi,yo)[1], 0, depth);

    Func subpout("subpout");
    Expr p = g(depth-1-cast(int32, abs(depth-2-p_clamped(yi,yo))), yi, yo);
    Expr n = g(cast(int32, abs(p_clamped(yi,yo)-1)), yi, yo);
    Expr d1 = p + n - 2*preout(yi,yo)[0] + abs(p - n);
    subpout(yi,yo) = (depth-preout(yi,yo)[1]-1+mindisp)*256 + (select(d1==0, 0, (p-n)*256/d1)+15)/16;

    g.bound(di,0,depth);
    

    Func output0("output0");
    output0(yi,yo)=select(textsum(yi,yo)<threshold, -1, subpout(yi,yo));



    Func output2("output2");
    output2(y)=output0(y%tilesize,y/tilesize);



    
    
    output0.compute_root();//.vectorize(yi,native_lanes);
    RVar ri("ri"), ro("ro");
    preout.update().split(rd, ro, ri, native_lanes);
    Var dii("dii");
    Func intermediate = preout.update().rfactor({{ri, dii}});

    intermediate.compute_at(output0, yo).vectorize(dii, native_lanes);
    intermediate.reorder(dii,yi,yo).reorder_storage(dii,yi,yo).update().reorder(dii, yi, ro).vectorize(dii, native_lanes);

    textsum.compute_at(output0, yo);
    subpout.compute_at(output0, yo);//.vectorize(yi, native_lanes);


    g.vectorize(di,native_lanes).update().vectorize(di,native_lanes);
    //g.update().reorder(yo);
    g.update(1).reorder(rx,yo).vectorize(di,native_lanes);

    g.compute_at(output0, yo);

    preout.update().reorder(yi,ri,yo);


    return output2;
}

bool diff_blur_1d_bad(){
    const int row = 1024*1024;
    const int acc = 1024;
    const int depth = 64; 
    const int winsize = 1;
    const int tilesize = 5;

    ImageParam A(Int(32), 1, "input0");
    ImageParam B(Int(32), 1, "input1");

    Func result = diff_blur_1d_inner_bad_full(A, B, row, winsize, depth, tilesize);

    Buffer<int32_t> a_buf(row);
    fill_buffer_1d_int(a_buf, row);
    A.set(a_buf);

    Buffer<int32_t> b_buf(row);
    fill_buffer_1d_int(b_buf, row);
    B.set(b_buf);

    Buffer<int32_t> out(row);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

Func diff_blur_2d_inner_bad_full(ImageParam &input0, ImageParam &input1, int winsize, int depth, int tilesize){
    Func output("output");
    Type int32 = Int(32);
    Type uint32 = UInt(32);
    Type int8 = Int(8);
    Type uint16 = UInt(16);

    const int native_lanes = 16;
    //const Expr depth = input.channels();
    Func f("f");
    Var x("x");
    Var y("y");
    Var d("d");
    Var di("di");
    Var xi("xi"), yi("yi"),xo("xo"),yo("yo");
    const int filterzero = 10;
    const int threshold = 1;
    const int mindisp = 0;

    int uniqueness_ratio = 15;

    Func diff("diff");

    Func b0("b0"), b1("b1");
    b0(y,x)=cast<int16_t>(BoundaryConditions::constant_exterior(input0, 0)(y,x));
    //b0.compute_root().vectorize(y,depth);
    b1(y,x)=cast<int16_t>(BoundaryConditions::constant_exterior(input1, 0)(y,x));
    //b1.compute_root().vectorize(y,depth);


    Func xsobel0("xsobel0"), xsobel1("xsobel1");
    Expr e0 = b0(y+1,x-1)-b0(y-1,x-1)+2*b0(y+1,x)-2*b0(y-1,x)+b0(y+1,x+1)-b0(y-1,x+1);
    Expr e1 = b1(y+1,x-1)-b1(y-1,x-1)+2*b1(y+1,x)-2*b1(y-1,x)+b1(y+1,x+1)-b1(y-1,x+1);

    const int fcap = 31;
    xsobel0(y,x)=cast<int16_t>(clamp(e0,-fcap,fcap)+fcap);
    xsobel1(y,x)=cast<int16_t>(clamp(e1,-fcap,fcap)+fcap);
    b0.compute_root().vectorize(y, native_lanes*8);
    b1.compute_root().vectorize(y, native_lanes*8);
    xsobel0.compute_root().vectorize(y, native_lanes*8);
    xsobel1.compute_root().vectorize(y, native_lanes*8);

    diff(di,yi,xi,yo,xo)=Halide::cast<uint8_t>(abs(xsobel0(yi+yo*tilesize,xi+xo*tilesize)-xsobel1(yi+yo*tilesize+di,xi+xo*tilesize)));

    RDom rx(1, tilesize-1, "rx");

    RDom rk(0,winsize,"rk");
    Func vsum("vsum"), g("g");
    vsum(di,yi,xi,yo,xo) = undef<uint16_t>();
    vsum(di,yi,0,yo,xo) = sum(cast<uint16_t>(diff(di,yi,rk,yo,xo)));
    vsum(di,yi,rx,yo,xo)=vsum(di,yi,rx-1,yo,xo)+diff(di,yi,rx+winsize-1,yo,xo)-diff(di,yi,rx-1,yo,xo);

    g(di,yi,xi,yo,xo) = undef<uint16_t>();
    g(di,0,xi,yo,xo) = sum(vsum(di,rk,xi,yo,xo));
    g(di,rx,xi,yo,xo) = g(di,rx-1,xi,yo,xo)+ vsum(di,rx+tilesize-1,xi,yo,xo)-vsum(di,rx-1,xi,yo,xo);
    Func text("text"), textsum("textsum"), textg("textg");
    text(yi,xi,yo,xo) = cast<uint16_t>(abs(b0(yi+yo*tilesize,xi+xo*tilesize)-filterzero));

    textsum(yi,xi,yo,xo) = undef<uint16_t>();
    textsum(yi,0,yo,xo) = sum(text(yi,rk,yo,xo));
    textsum(yi,rx,yo,xo) = textsum(yi,rx-1,yo,xo)+ text(yi,rx+winsize-1,yo,xo)-text(yi,rx-1,yo,xo);

    textg(yi,xi,yo,xo) = undef<uint16_t>();
    textg(0,xi,yo,xo) = sum(textsum(rk,xi,yo,xo));
    textg(rx,xi,yo,xo) = textg(rx-1,xi,yo,xo)+ textsum(rx+tilesize-1,xi,yo,xo)-textsum(rx-1,xi,yo,xo);

    Func preout("preout");

    RDom rd(0,depth,"rd");

    /*preout(yi,xi,yo,xo)=select(preout(yi,xi,yo,xo)[0]<g(rd,yi,xi,yo,xo), preout(yi,xi,yo,xo), Tuple(g(rd,yi,xi,yo,xo), rd));

    Func p_clamped("p_clamped");
    p_clamped(yi,xi,yo,xo) = unsafe_promise_clamped(preout(yi,xi,yo,xo)[1], 0, depth);

    Func subpout("subpout");
    Expr p = g(depth-1-cast(int32, abs(depth-2-p_clamped(yi,xi,yo,xo))), yi, xi, yo, xo);
    Expr n = g(cast(int32, abs(p_clamped(yi,xi,yo,xo)-1)), yi, xi, yo, xo);
    Expr d1 = p + n - 2*preout(yi,xi,yo,xo)[0] + abs(p - n);
    subpout(yi,xi,yo,xo) = preout(yi,xi,yo,xo)[1];//(depth-preout(yi,xi,yo,xo)[1]-1+mindisp)*256 + (select(d1==0, 0, (p-n)*256/d1)+15)/16;
    
    preout(yi,x,yo)=min(blur_y(rd,yi,x,yo),preout(yi,x,yo));
    preout.update().atomic(false).vectorize(rd, depth);*/

    preout(yi,xi,yo,xo)=cast<uint16_t>(65535);//Tuple(cast<uint16_t>(65535), cast<uint16_t>(0));

    //preout(yi,x,yo)=Tuple(blur_y(0,yi,x,yo)+blur_y(depth-1,yi,x,yo)+vsum(depth-1,yi,x,yo)+vsum(0,yi,x,yo), cast<uint16_t>(2));//select(preout(yi,x,yo)[0]<vsum(rd,yi,x,yo), preout(yi,x,yo), Tuple(vsum(rd,yi,x,yo), cast<uint16_t>(rd)));

    preout(yi,xi,yo,xo)=min(g(rd,yi,xi,yo,xo),preout(yi,xi,yo,xo));
    debug(1)<<"b";
    preout.update().atomic(false).vectorize(rd, depth);
    debug(1)<<"c";

    Func prearg("prearg");
    prearg(di, yi, xi, yo, xo) = select(preout(yi,xi,yo,xo)==g(di,yi,xi,yo,xo), cast<uint16_t>(di), cast<uint16_t>(65535));
    Func argmin1("argmin");
    argmin1(yi,xi,yo,xo)=cast<uint16_t>(65535);
    debug(1)<<"d";
    argmin1(yi,xi,yo,xo)=min(argmin1(yi,xi,yo,xo), prearg(rd, yi, xi, yo, xo));
    debug(1)<<"x";
    argmin1.update().atomic(false).vectorize(rd, depth);
    debug(1)<<"y";


    Func second_best("second_best");
    second_best(di,yi,xi,yo,xo) = select(abs(di-argmin1(yi,xi,yo,xo))<=1,cast<uint16_t>(65535), g(di,yi,xi,yo,xo));
    debug(1)<<"z";
    Func argmin2("argmin2");
    debug(1)<<"e";
    argmin2(yi,xi,yo,xo)=cast<uint16_t>(65535);
    argmin2(yi,xi,yo,xo)=min(argmin2(yi,xi,yo,xo), second_best(rd, yi, xi, yo, xo));
    argmin2.update().atomic(false).vectorize(rd, depth);
    debug(1)<<"??";

    Func p_clamped("p_clamped");
    p_clamped(yi,xi,yo,xo) = unsafe_promise_clamped(argmin1(yi,xi,yo,xo), 1, depth-2);
    Func subpout(int32, "subpout");
    Expr p = cast<int16_t>(g(cast(int32,p_clamped(yi,xi,yo,xo))+1, yi, xi, yo, xo));
    Expr n = cast<int16_t>(g(cast(int32,p_clamped(yi,xi,yo,xo))-1, yi, xi, yo, xo));
    Expr d1 = p + n - 2*preout(yi,xi,yo,xo) + abs(p - n);
    debug(1)<<"d1 defined\n";
    Expr subpout_expr = cast<int32_t>(cast<int16_t>(depth-p_clamped(yi,xi,yo,xo)-1+mindisp)*256 + (select(d1==0, 0, (p-n)*256/d1)+15)<<4);
    subpout(yi,xi,yo,xo) = select(argmin1(yi,xi,yo,xo)>0&&argmin1(yi,xi,yo,xo)<depth-1, subpout_expr, cast<int32_t>(depth-p_clamped(yi,xi,yo,xo)-1+mindisp)*256);
    debug(1)<<"subpout defined\n";

    g.bound(di,0,depth);
    vsum.bound(di,0,depth);
    

    Func output0("output0");
    output0(yi,xi,yo,xo)=select(textg(yi,xi,yo,xo)<threshold||argmin2(yi,xi,yo,xo)<=preout(yi,xi,yo,xo)*cast<uint16_t>(100+uniqueness_ratio)/100, -1, subpout(yi,xi,yo,xo));
    
    g.bound(di,0,depth);
    

    //Func output0("output0");
    //output0(yi,xi,yo,xo)=select(textg(yi,xi,yo,xo)<threshold, -1, subpout(yi,xi,yo,xo));



    Func output2("output2");
    output2(y,x)=output0(y%tilesize,x%tilesize,y/tilesize,x/tilesize);

    
    output0.compute_root();//.vectorize(yi,native_lanes);
    RVar ri("ri"), ro("ro");
    //preout.update().split(rd, ro, ri, native_lanes);
    //Var dii("dii");
    //Func intermediate = preout.update().rfactor({{ri, dii}});

    //intermediate.compute_at(output0, yo).vectorize(dii, native_lanes);
    //intermediate.reorder(dii,yi,xi,yo,xo).reorder_storage(dii,yi,xi,yo,xo).update().reorder(dii, yi,xi, ro).vectorize(dii, native_lanes);

    textsum.compute_at(output0, yo);
    textg.compute_at(output0, yo);
    //subpout.compute_at(output0, yo);//.vectorize(yi, native_lanes);
    preout.compute_at(output0, yi);
    argmin1.compute_at(output0, yi);
    argmin2.compute_at(output0, yi);


    g.vectorize(di,native_lanes).update().vectorize(di,native_lanes);
    //g.update().reorder(yo);
    g.update(1).reorder(rx,yo,xi,xo).vectorize(di,native_lanes);

    vsum.compute_at(output0, yo).reorder(di,yi,xi,yo,xo).vectorize(di,native_lanes).update().reorder(di,yi,yo,xo).vectorize(di,native_lanes);
    vsum.update(1).reorder(di,yi,rx,yo,xo).vectorize(di,native_lanes);
    g.compute_at(output0, yo);

    //preout.update().reorder(yi,xi,ri,yo,xo);

    return output2;
}


Func diff_blur_2d_inner_good_full(ImageParam &input0, ImageParam &input1, int winsize, int depth, int tilesize){
     Func output("output");
    Type int8 = Int(8);
    Type uint16 = UInt(16);
    Type int32 = Int(32);
    const int native_lanes = 16;
    //const Expr depth = input.channels();
    Func f("f");
    Var x("x");
    Var y("y");
    Var d("d");
    Var di("di");
    Var xi("xi"), yi("yi"),xo("xo"),yo("yo");
    const int filterzero = 10;
    const int threshold = 1;
    const int mindisp = 0;

    int uniqueness_ratio = 15;

    Func diff("diff");

    Func b0("b0"), b1("b1");
    b0(y,x)=cast<int16_t>(BoundaryConditions::constant_exterior(input0, 0)(y,x));
    //b0.compute_root().vectorize(y,depth);
    b1(y,x)=cast<int16_t>(BoundaryConditions::constant_exterior(input1, 0)(y,x));
    //b1.compute_root().vectorize(y,depth);


    Func xsobel0("xsobel0"), xsobel1("xsobel1");
    Expr e0 = b0(y+1,x-1)-b0(y-1,x-1)+2*b0(y+1,x)-2*b0(y-1,x)+b0(y+1,x+1)-b0(y-1,x+1);
    Expr e1 = b1(y+1,x-1)-b1(y-1,x-1)+2*b1(y+1,x)-2*b1(y-1,x)+b1(y+1,x+1)-b1(y-1,x+1);

    const int fcap = 31;
    xsobel0(y,x)=cast<int16_t>(clamp(e0,-fcap,fcap)+fcap);
    xsobel1(y,x)=cast<int16_t>(clamp(e1,-fcap,fcap)+fcap);
    b0.compute_root().vectorize(y, native_lanes*8);
    b1.compute_root().vectorize(y, native_lanes*8);
    xsobel0.compute_root().vectorize(y, native_lanes*8);
    xsobel1.compute_root().vectorize(y, native_lanes*8);


    
    diff(di,yi,x,yo)=Halide::cast<uint16_t>(cast<uint16_t>(abs((xsobel0(yi+yo*tilesize,x))-(xsobel1(yi+yo*tilesize+di-depth+1,x)))));
    Func vsum(uint16, "vsum");
    Func zero_blur(uint16,"zero_blur");

    RDom rx0(0, winsize);

    zero_blur(di, yi, yo) = sum(cast<uint16_t>(diff(di, yi, rx0, yo)));
    vsum(di,yi,x,yo) = select(x<=0, zero_blur(di, yi, yo), likely(vsum(di, yi, max(x-1,0), yo) + diff(di, yi, x+winsize-1, yo) - diff(di, yi, max(x-1, 0), yo))); /// should be x-1
    //vsum(di,yi,x,yo) = select(x<=0, zero_blur(di, yi, yo),likely(vsum(di, yi, max(x-1,0), yo)+cast<uint8_t>(b0(yi+yo*tilesize+di,x))));
    debug(1)<<"vsum defined\n";
    Func blur_y(uint16, "blur_y");
    RDom ry(0, tilesize, "ry");
    RDom ry1(1, depth-1, "ry1");
    //blur_y(di,yi,x,yo) = undef<uint16_t>();
    Func f1("f1");
    f1(di, x, yo) = sum(vsum(di,rx0, x, yo));
    //blur_y(di, 0, x, yo) = f1(di, x, yo);
    //blur_y(di,ry1,x,yo) = blur_y(di, ry1-1, x, yo) + vsum(di, ry1+winsize-1, x, yo) - vsum(di, ry1-1, x, yo);
    
    //blur_y(di,ry,x,yo) = select(ry == 0, f1(di, x, yo), likely(blur_y(di, max(ry - 1, 0), x, yo) + vsum(di, ry + winsize - 1, x, yo) -vsum(di, max(ry - 1, 0), x, yo)));
    
    
    blur_y(di,yi,x,yo) = select(yi<=0, f1(di, x, yo), likely(blur_y(di, max(yi-1,0), x, yo) + vsum(di, yi+winsize-1, x, yo) - vsum(di, max(yi-1,0), x, yo)));
    //should be ry-1

    debug(1)<<"blur_y defined\n";

    Func text("text"), zerotext("zerotext"), textsum( "textsum"),textf1("textf1"), textblury("textblury");
    text(yi,x,yo) = cast<uint8_t>(abs(cast<int8_t>(b0(yi+yo*tilesize,x))-cast<int8_t>(filterzero)));
    zerotext(yi,yo) = sum(cast<uint16_t>(text(yi,rx0,yo)));
    textsum(yi,x,yo) = select(x<=0, zerotext(yi,yo), textsum(yi,x-1,yo)+ text(yi,x+winsize-1,yo)-text(yi,x-1,yo));
    textf1(x,yo)=sum(textsum(rx0,x,yo));
    textblury(yi,x,yo) = undef<uint16_t>();
    textblury(ry,x,yo) = select(ry==0, textf1(x,yo), textblury(max(ry-1,0),x,yo)+textsum(ry+winsize-1,x,yo)-textsum(max(ry-1,0),x,yo));

    Func preout("preout");

    RDom rd(0,depth,"rd");
    preout(yi,x,yo)=cast<uint16_t>(65535);//Tuple(cast<uint16_t>(65535), cast<uint16_t>(0));

    //preout(yi,x,yo)=Tuple(blur_y(0,yi,x,yo)+blur_y(depth-1,yi,x,yo)+vsum(depth-1,yi,x,yo)+vsum(0,yi,x,yo), cast<uint16_t>(2));//select(preout(yi,x,yo)[0]<vsum(rd,yi,x,yo), preout(yi,x,yo), Tuple(vsum(rd,yi,x,yo), cast<uint16_t>(rd)));

    preout(yi,x,yo)=min(blur_y(rd,yi,x,yo),preout(yi,x,yo));
    preout.update().atomic(false).vectorize(rd, depth);


    Func prearg("prearg");
    prearg(di, yi, x, yo) = select(preout(yi,x,yo)==blur_y(di,yi,x,yo), cast<uint16_t>(di), cast<uint16_t>(65535));
    Func argmin1("argmin");
    argmin1(yi,x,yo)=cast<uint16_t>(65535);
    argmin1(yi,x,yo)=min(argmin1(yi,x,yo), prearg(rd, yi, x, yo));
    argmin1.update().atomic(false).vectorize(rd, depth);


    Func second_best("second_best");
    second_best(di,yi,x,yo) = select(abs(di-cast<int16_t>(argmin1(yi,x,yo)))<=1,cast<uint16_t>(65535), blur_y(di,yi,x,yo));
    Func argmin2("argmin2");
    argmin2(yi,x,yo)=cast<uint16_t>(65535);
    argmin2(yi,x,yo)=min(argmin2(yi,x,yo), second_best(rd, yi, x, yo));
    argmin2.update().atomic(false).vectorize(rd, depth);

    Func p_clamped("p_clamped");
    p_clamped(yi,x,yo) = unsafe_promise_clamped(argmin1(yi,x,yo), 1, depth-2);

    Func subpout(int32, "subpout");
    Expr p = cast<int32_t>(blur_y(cast(int32,p_clamped(yi,x,yo))+1, yi, x, yo));
    Expr n = cast<int32_t>(blur_y(cast(int32,p_clamped(yi,x,yo))-1, yi, x, yo));
    Expr d1 = p + n - 2*preout(yi,x,yo) + abs(p - n);
    debug(1)<<"d1 defined\n";
    Expr subpout_expr = cast<int32_t>((cast<int16_t>(depth-p_clamped(yi,x,yo)-1+mindisp)*256 + (select(d1==0, 0, (p-n)*256/d1)+15))>>4);
    subpout(yi,x,yo) = select(argmin1(yi,x,yo)>0&&argmin1(yi,x,yo)<depth-1, subpout_expr, cast<int32_t>(depth-argmin1(yi,x,yo)-1+mindisp)*16);
    debug(1)<<"subpout defined\n";

    blur_y.bound(di,0,depth);
    vsum.bound(di,0,depth);
    

    Func output0("output0");
    output0(yi,x,yo)=select((textblury(yi,x,yo)<threshold||cast<int32_t>(argmin2(yi,x,yo))<=cast<int32_t>(preout(yi,x,yo))+(cast<int32_t>(preout(yi,x,yo))*cast<int32_t>(uniqueness_ratio))/100), 0, subpout(yi,x,yo));

    argmin2.compute_at(output0, yi);
    argmin1.compute_at(output0, yi);
    preout.compute_at(output0, yi);

    Func output2("output2");
    output2(y,x)=output0(y%tilesize,x,y/tilesize);
    
    
    output0.compute_root().reorder_storage(yi,yo,x);//.parallel(yo);
    RVar ri("ri"), ro("ro");
    //preout.compute_at(output0, x);
    //preout.update().split(rd, ro, ri, native_lanes);
    Var dii("dii");
    // print out output.update dimensions:
    //Func intermediate = preout.update().rfactor({{ri, dii}});
   

    //intermediate.compute_at(output0, x);
    //intermediate.reorder(dii,yi,x,yo).vectorize(dii,native_lanes).reorder_storage(dii,yi,x,yo).update().reorder(dii, yi, x, ro).vectorize(dii, native_lanes);



    //blur_y.compute_at(output0,x).update().reorder(di,ry).vectorize(di,native_lanes*2);
    blur_y.compute_at(output0,x).store_at(output0, x).vectorize(di,depth).fold_storage(yi,2);
    //blur_y.update(1).reorder(di,ry1).vectorize(di,native_lanes*4);
    vsum.compute_at(output0,x).store_at(output0, yo).vectorize(di,depth).fold_storage(x,2);

    f1.compute_at(output0,x).vectorize(di,depth);
    zero_blur.compute_at(output0, yo).vectorize(di, depth);

    zerotext.compute_at(output0, yo);
    textf1.compute_at(output0, x);
    textsum.compute_at(output0, x).store_at(output0, yo).fold_storage(x,2);
    textblury.compute_at(output0, x);
    //subpout.compute_at(output0, x);

    output2.vectorize(y,native_lanes);

    return output2;

}



Func diff_blur_2d_inner_naive(ImageParam &input0, ImageParam &input1, int winsize, int depth, int tilesize){
     Func output("output");
    Type int8 = Int(8);
    Type uint16 = UInt(16);
    Type int32 = Int(32);
    const int native_lanes = 16;
    //const Expr depth = input.channels();
    Func f("f");
    Var x("x");
    Var y("y");
    Var d("d");
    Var di("di");
    Var xi("xi"), yi("yi"),xo("xo"),yo("yo");
    const int filterzero = 10;
    const int threshold = 0;
    const int mindisp = 0;

    Func diff("diff");

    Func b0("b0"), b1("b1");
    b0(y,x)=cast<int16_t>(BoundaryConditions::constant_exterior(input0, 0)(y,x));
    //b0.compute_root().vectorize(y,depth);
    b1(y,x)=cast<int16_t>(BoundaryConditions::constant_exterior(input1, 0)(y,x));
    //b1.compute_root().vectorize(y,depth);


    Func xsobel0("xsobel0"), xsobel1("xsobel1");
    Expr e0 = b0(y+1,x-1)-b0(y-1,x-1)+2*b0(y+1,x)-2*b0(y-1,x)+b0(y+1,x+1)-b0(y-1,x+1);
    Expr e1 = b1(y+1,x-1)-b1(y-1,x-1)+2*b1(y+1,x)-2*b1(y-1,x)+b1(y+1,x+1)-b1(y-1,x+1);

    const int fcap = 31;
    xsobel0(y,x)=clamp(e0,-fcap,fcap)+fcap;
    xsobel1(y,x)=clamp(e1,-fcap,fcap)+fcap;
    xsobel0.compute_root().vectorize(y, native_lanes*4);
    xsobel1.compute_root().vectorize(y, native_lanes*4);



    
    diff(di,y,x)=Halide::cast<uint16_t>(cast<int16_t>(abs(cast<int16_t>(xsobel0(y,x))-cast<int16_t>(xsobel1(y+(di-depth+1),x)))));
    Func blur(uint16, "blur");
    blur(di,y,x)=cast<uint16_t>(0);
    RDom r(-(winsize/2),winsize,-(winsize/2),winsize,"r");
    blur(di,y,x)+=diff(di,r.y+y,r.x+x);
    Func min1(uint16, "min1");
    min1(y,x)=cast<uint16_t>(65535);
    RDom r2(0,depth);
    min1(y,x)=min(min1(y,x), blur(r2,y,x));
    Func argmin1("argmin1");
    argmin1(y,x)=cast<int32_t>(65535);
    argmin1(y,x)=select(min1(y,x)==blur(r2,y,x)&&argmin1(y,x)==65535, (depth-cast<int32_t>(r2)-1)*16, argmin1(y,x));
    blur.compute_root().parallel(y).update().parallel(y);
    min1.compute_root().parallel(y).update().parallel(y);
    argmin1.compute_root().parallel(y).update().parallel(y);
    return xsobel0;

}


bool diff_blur_2d_good_full(){
    
    const int depth = 64*2; 
    const int winsize = 15;
    const int tilesize = 64;

    ImageParam A(UInt(8), 2, "input0");
    ImageParam B(UInt(8), 2, "input1");

    Halide::Buffer<uint8_t> left = load_image("../stereo_test/aloeL.jpg");
    Halide::Buffer<uint8_t> right = load_image("../stereo_test/aloeR.jpg");
    Halide::Buffer<uint8_t> left_gray(left.width(), left.height());
    Halide::Buffer<uint8_t> right_gray(right.width(), right.height());
    for(int i=0; i<left.width(); i++){
        for(int j=0; j<left.height(); j++){
            int r = left(i,j,0);
            int g = left(i,j,1);
            int b = left(i,j,2);
            left_gray(i,j) = (uint8_t)((b*1868 + g*9617 + r*4899 + 8192) >> 14);
        }
    }
    for(int i=0; i<right.width(); i++){
        for(int j=0; j<right.height(); j++){
            int r = right(i,j,0);
            int g = right(i,j,1);
            int b = right(i,j,2);
            right_gray(i,j) = (uint8_t)((b*1868 + g*9617 + r*4899 + 8192) >> 14);
        }
    }
    A.set(left_gray);
    B.set(right_gray);

    int row = left.width();
    int acc = left.height();

    Func result = diff_blur_2d_inner_good_full(A, B, winsize, depth, tilesize);

    

    //Buffer<uint8_t> a_buf(row,acc);
    //fill_buffer_2d_int8(a_buf, row, acc);
    //A.set(a_buf);

    //Buffer<uint8_t> b_buf(row,acc);
    //fill_buffer_2d_int8(b_buf, row, acc);
    //B.set(b_buf);

    Buffer<int32_t> out(row, acc);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out);
    });

    Halide::Buffer<uint8_t> out_u8(row, acc,3);
    int out_max=out(0,0)/16;
    // convert to 3-channel
    for(int i=0; i<row; i++){
        for(int j=0; j<acc; j++){
            uint8_t val = (uint8_t)(out(i,j)/8);
            out_max=std::max((int)val, out_max);
            out_u8(i,j,0) = val;
            out_u8(i,j,1) = val;
            out_u8(i,j,2) = val;
        }
    }

    std::cout<<out_max<<out(100,100)<<std::endl;
        


    save_image(out_u8, "../stereo_test/lsobelhal.png");

    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";

    Halide::Buffer<uint8_t> out_left(row, acc,3);
     //convert to 3-channel
    for(int i=0; i<row; i++){
        for(int j=0; j<acc; j++){
            uint8_t val = left_gray(i,j)*99;
            out_left(i,j,0) = val;
            out_left(i,j,1) = val;
            out_left(i,j,2) = val;
        }
    }
    save_image(out_left, "../stereo_test/leftgrayhal.png");

    return true;
}



bool diff_blur_2d_bad_full(){
    const int row = 1024;
    const int acc = 1024;
    const int depth = 64*2; 
    const int winsize = 15;
    const int tilesize = 64;

    ImageParam A(Int(32), 2, "input0");
    ImageParam B(Int(32), 2, "input1");

    Func result = diff_blur_2d_inner_bad_full(A, B, winsize, depth, tilesize);

    Buffer<int32_t> a_buf(row,acc);
    fill_buffer_2d_int(a_buf, row, acc);
    A.set(a_buf);

    Buffer<int32_t> b_buf(row,acc);
    fill_buffer_2d_int(b_buf, row, acc);
    B.set(b_buf);

    Buffer<int32_t> out(row,acc);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}


Func merge_bad_inner(ImageParam &A, ImageParam &B, int lena, int lenb){
    // needs to append inf at the ends
    Func MergeIndex("MergeIndex"); // index of the LHS after x elements are merged
    Var x("x");
    RDom r(1, lena+lenb-1, "r");
    MergeIndex(x) = undef<int>();
    MergeIndex(0) = 0;
    //MergeIndex(r) = select(r-1-MergeIndex(r-1)==lenb||A(unsafe_promise_clamped(MergeIndex(r-1), 0, lena))<B(unsafe_promise_clamped(r-1-MergeIndex(r-1), 0, lenb)), MergeIndex(r-1)+1, MergeIndex(r-1));
    MergeIndex(r) = select(A(unsafe_promise_clamped(MergeIndex(r-1), 0, lena))<B(unsafe_promise_clamped(r-1-MergeIndex(r-1), 0, lenb)), MergeIndex(r-1)+1, MergeIndex(r-1));

    MergeIndex(lena+lenb)=lena;
    RDom r2(0, lena+lenb, "r2");
    Func out("out");
    out(x) = undef<int>();
    out(r) = select(MergeIndex(r)==MergeIndex(r+1), B(unsafe_promise_clamped(r-MergeIndex(r), 0, lenb)), A(unsafe_promise_clamped(MergeIndex(r), 0, lena)));
    out(0) = min(A(0), B(0));
    MergeIndex.compute_root();
    return out;
}   


Func merge_good_inner(ImageParam &A, ImageParam &B, int lena, int lenb){
    // needs to append inf at the ends
    Func MergeIndex(Int(32), "MergeIndex"); // index of the LHS after x elements are merged
    Var x("x");
    RDom r(1, lena+lenb-1, "r");
    //MergeIndex(x) = select(x<=0, 0, select(x-1-MergeIndex(x-1)==lenb||A(unsafe_promise_clamped(MergeIndex(x-1), 0, lena))<B(unsafe_promise_clamped(x-1-MergeIndex(x-1), 0, lenb)), MergeIndex(x-1)+1, MergeIndex(x-1)));
    MergeIndex(x) = select(x<=0, 0, select(A(unsafe_promise_clamped(MergeIndex(x-1), 0, lena))<B(unsafe_promise_clamped(x-1-MergeIndex(x-1), 0, lenb)), MergeIndex(x-1)+1, MergeIndex(x-1)));

    RDom r2(0, lena+lenb, "r2");
    Func out("out");
    out(x) = undef<int>();
    out(r) = select(A(unsafe_promise_clamped(MergeIndex(r), 0, lena))>=B(unsafe_promise_clamped(r-MergeIndex(r), 0, lenb)), B(unsafe_promise_clamped(r-MergeIndex(r), 0, lenb)), A(unsafe_promise_clamped(MergeIndex(r), 0, lena)));
    MergeIndex.compute_at(out, r).store_root();
    return out;
}   

Func merge_good_inner_parallel(ImageParam &A, ImageParam &B, int lena, int nproc){
    // needs to append inf at the ends
    Func MergeIndex(Int(32), "MergeIndex"); // index of the LHS after x elements are merged
    Func premerge("premerge");

    Var x("x");

    Var xi("xi"), xo("xo"), k("k");

    int sublen = 2*lena/nproc;
    RDom r0(1, static_cast<int>(std::log2(lena))+4, "r0");

    premerge(k, xo) = Tuple(undef<int>(), undef<int>()); // when does it cross from < to >=
    premerge(0, xo) = Tuple(min(max(0, xo*sublen-lena), lena), min(xo*sublen, lena));
    Expr midx = (premerge(r0 - 1, xo)[0]+premerge(r0 - 1, xo)[1])/2;
    Expr bbaseidx = xo*sublen-1;
    premerge(r0, xo) = select(A(unsafe_promise_clamped(midx, 0, lena))>=B(unsafe_promise_clamped(bbaseidx-midx, 0, lena)), Tuple(premerge(r0 - 1, xo)[0], midx), Tuple(midx, premerge(r0 - 1, xo)[1]));
    premerge(r0, 0) = Tuple(0, 0);
    premerge(r0, nproc) = Tuple(lena, lena);
    Func segmerge("segmerge");

    Expr idx2 = premerge(static_cast<int>(std::log2(sublen))+4, xo)[0];
    segmerge(xo) = select(A(unsafe_promise_clamped(idx2, 0, lena))<B(unsafe_promise_clamped(bbaseidx-idx2, 0, lena)), premerge(static_cast<int>(std::log2(sublen))+4, xo)[1], premerge(static_cast<int>(std::log2(sublen))+4, xo)[0]);

    Func segidx(Int(32), "segidx");
    Expr x2 = xi+sublen*xo;
    segidx(xi, xo) = select(xi<=0, segmerge(xo), likely(select(x2-1-segidx(max(xi-1,0),xo)==lena||A(clamp(segidx(max(xi-1,0),xo), 0, lena))<B(clamp(x2-1-segidx(max(xi-1,0),xo), 0, lena)), segidx(max(xi-1,0),xo)+1, segidx(max(xi-1,0),xo))));
    //segidx(x) = 9999999;
    //RDom r3(0, 20, "r3");
    //segidx(r3) = select(r3<=0, 0, likely(select((A(segidx(0))<4), 1, 0)));
    Func test("test");
    Func z("z");
    z(x) = undef<int>();
    z(0) = 1;
    Func segout("segout");
    segout(xi, xo) = undef<int>();
    RDom r(0, sublen, "r");
    r.where(xo*sublen+r<2*lena);
    segout(r, xo) = select(A(unsafe_promise_clamped(segidx(r, xo), 0, lena))>=B(unsafe_promise_clamped(xo*sublen+r-segidx(r, xo), 0, lena)), B(unsafe_promise_clamped(xo*sublen+r-segidx(r, xo), 0, lena)), A(unsafe_promise_clamped(segidx(r, xo), 0, lena)));


    Func out("out");
    out(x) = segout(x%sublen, x/sublen);
    premerge.compute_root();
    segmerge.compute_root();
    segidx.compute_at(segout, r).store_at(segout, xo);
    segout.compute_root().parallel(xo).update().parallel(xo).vectorize(r, 8);
    out.vectorize(x, 8);
    //out(x) = premerge((x/1)%static_cast<int>(std::log2(sublen)+2), (x/1)/static_cast<int>(std::log2(sublen)+2))[0];
    //segidx.compute_root();
    //segmerge.compute_root();
    return out;
}   

bool MergeBad(){
    const int row = 5000000;


    ImageParam A(Int(32), 1, "input0");
    ImageParam B(Int(32), 1, "input1");

    Func result = merge_good_inner(A, B, row, row);//merge_good_inner_parallel(A, B, row, 8);

    Buffer<int32_t> a_buf(row+1);
    a_buf(0)=0;
    for(int i=1;i<row;i++){
        a_buf(i) = a_buf(i-1)+static_cast<int32_t>(((float)rand() / (float)(RAND_MAX)) * 5.f);
    }
    a_buf(row)=100000000000;
    A.set(a_buf);


    Buffer<int32_t> b_buf(row+1);
    b_buf(0)=0;
    for(int i=1;i<row;i++){
        b_buf(i) = b_buf(i-1)+static_cast<int32_t>(((float)rand() / (float)(RAND_MAX)) * 5.f);
    }
    b_buf(row)=100000000000;
    B.set(b_buf);


    Buffer<int32_t> out(10000000);
    auto time = Tools::benchmark(4, 4, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}


bool bounds_test_1(){

    Func output("output"), intermediate("intermediate"), B("B");
    Var x("x");

    B(x) = x%5;
    
    Buffer<int32_t> a_buf(1024,1024);
    fill_buffer_2d_int(a_buf, 1024,1024);
    ImageParam input(Int(32), 2, "input0");
    input.set(a_buf);


    Func C("C");
    Func argminfunc("argminfunc");

    //intermediate(x) = x*x;
    //intermediate(r) += r;

    RDom r(0, 100);
    C(x)=argmin(input(r, x), argminfunc);
    output(x)=input(cast<int>(abs(unsafe_promise_clamped(C(x)[0], 0, 100)-6)), 0);

    C.compute_root().bound(x,0,100);
    
    //intermediate.compute_root();
    Buffer<int32_t> out(1);
    output.realize(out);
    auto time = Tools::benchmark(5, 5, [&]() {
        output.realize(out);    
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

bool hist_test(){
    Buffer<int32_t> a_buf(1024);
    fill_buffer_1d_int(a_buf, 1024);
    ImageParam input(Int(32), 1, "input0");
    input.set(a_buf);

    int hist_min = 0;
    int hist_max = 255;
    int num_bins = 256;
    Var x("x");
    Func hist("hist");
    RDom r(0, input.dim(0).extent());
    Expr hist_index = cast<int>(num_bins * ((input(r) - hist_min) / (hist_max - hist_min)));
    hist(hist_index) += 1;

    Buffer<int32_t> out(num_bins);
    auto time = Tools::benchmark(5, 5, [&]() {
        hist.realize(out);    
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}

bool argmin_rfactor_test(){
    const int length = 1<<10;

    Func input("input");
    Var x("x");
    input(x) = (x-5)*(x-5);
    Func output("output");

    RDom r(0, 50);
    Func argminfunc("argminfunc");
    output() = argmax(input(r), argminfunc);

    RVar ri("ri"), ro("ro");
    argminfunc.update().split(r, ro, ri, 8);
    Var xi("xi");


    // print update stage values
    //std::cout<<argminfunc.update().get_definition().values()<<std::endl;
    Func intermediate = argminfunc.update().rfactor({{ro, xi}});



    Buffer<int> out(1);
    auto time = Tools::benchmark(5,5, [&]() {
        output.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;

}












int main(int argc, char **argv) {
    Target target = get_jit_target_from_environment();
    //argmin_rfactor_test();
    printf("Running normal IIR...\n");
    //iir_normal();
    //box_blur_s2();
    printf("Running inductive IIR...\n");
    //iir_inductive();

    //bounds_test_1();
    //hist_test();
    //diff_blur_1d_good();
    //diff_blur_1d_bad();


    //MergeBad();
    //diff_blur_2d_good_full();
    //diff_blur_2d_good_full();
    //sum_inductive();

    sum_normal();
    sum_inductive();
    sum_normal();
    sum_inductive();


    //diff_blur_2d_bad_full();
    //diff_blur_s2();
    //diff_blur_s1();
    //diff_blur_s1();
    //diff_blur_s2();
    printf("Running normal box blur...\n");
    //box_blur_normal();
    printf("Running normal sum scan...\n");
    //sum_normal();
    printf("Running inductive sum scan...\n");
    //sum_inductive();
    return 0;
}