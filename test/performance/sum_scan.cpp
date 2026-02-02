#include "Halide.h"
#include "halide_benchmark.h"
#include "halide_test_dirs.h"

#include <cstdint>
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


Func sum_scan_inductive(Func input, Expr length) {
    Func f4 = Func(Int(32), "f4");
    Func f1("f1"), f2("f2"), f3("f3"), f5("f5"), in("in");
    Var x("x"), xo("xo"), xi("xi");
    in(x) = x + 1;
    f1(x) = input(x) + input(x + 3);
    f2(x) = f1(x) + f1(x + 2);
    f3(x) = f2(x) + f2(x + 2);
    f4(x) = select(x < 8, 0, likely(f3(x) + f4(x-8)));
    f5(x) = f4(x) / 4;
    f5.bound(x, 0, length).split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
    //f4.bound(x, -8, 1024);
    //f3.bound(x, 0, 1025);
    //f2.bound(x, 0, 1026);
    //f1.bound(x, 0, 1028);
    
    //f5.split(x, xo, xi, 8, TailStrategy::RoundUp).vectorize(xi);
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


Func sum_scan_normal(Func input, Expr length) {
    Func f4 = Func(Int(32), "f4");
    Func f1("f1"), f2("f2"), f3("f3"), f5("f5"), in("in");
    Var x("x"), xo("xo"), xi("xi");
    in(x) = x + 1;
    f1(x) = input(x) + input(x + 4);
    f2(x) = f1(x) + f1(x + 2);
    f3(x) = f2(x) + f2(x + 1);
    f4(x) = select(x < 8, f3(x)-f3(x), likely(f3(x) + f4(x-8)));
    f5(x) = f4(x) / 4;
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

    Buffer<int> a_buf(length + 8);
    for (int i = 0; i < length + 8; ++i) {
        a_buf(i) = i + 1;
    }
    A.set(a_buf);

    Buffer<int> out(length);
    auto time = Tools::benchmark(5,5, [&]() {
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

    Buffer<int> a_buf(length + 8);
    for (int i = 0; i < length + 8; ++i) {
        a_buf(i) = i + 1;
    }
    A.set(a_buf);

    Buffer<int> out(length);
    auto time = Tools::benchmark(5, 5, [&]() {
        result.realize(out);
    });
    std::cout << "Exec time: " << time << "\n";
    std::cout << "Success!\n";
    return true;
}




Func diff_blur_good(Func input0, int height, int winsize, int depth, int tilesize){
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


    diff(di,d,yi,x,yo)=abs(input0(yi+yo*tilesize,x)-input0(yi+yo*tilesize+di+d*native_lanes,x));
    Func vsum("vsum");
    Func zero_blur("zero_blur");

    RDom rx0(0, winsize);

    zero_blur(di, d, yi, yo) = sum(diff(di, d, yi, rx0, yo));
    vsum(di,d,yi,x,yo) = select(x<=0, zero_blur(di, d,yi, yo), likely(vsum(di, d,yi, x-1, yo) + diff(di, d,yi, x+winsize-1, yo) - diff(di, d, yi, x-1, yo))); /// should be x-1
    
    Func blur_y("blur_y");
    RDom ry(0, height-winsize, "ry");
    blur_y(di,d,yi,x,yo) = undef<float>();
    Func f1("f1");
    f1(di, x, yo, d) = sum(vsum(di, d,rx0, x, yo));
    blur_y(di,d,ry,x,yo) = select(ry == 0, f1(di, d, x, yo), likely(blur_y(di, d, ry - 1, x, yo) + vsum(di, d, ry + winsize - 1, x, yo) -vsum(di, d,ry - 1, x, yo)));
    //should be ry-1

    Func preout("preout");

    RDom rdi(0, depth/native_lanes, "rdi");
    Func argminfunc;

    preout(di,yi,x,yo)=argmin(blur_y(di,rdi,yi,x,yo), argminfunc);

    RDom rd(0,native_lanes,"rd");
    Func argminfunc2;
    Func out1("out1");
    out1(yi,x,yo)=0;//min(preout(rd,yi,x,yo), argminfunc2)[0];
    output(y,x)=out1(y%tilesize,x,y/tilesize);

    argminfunc2.compute_root().update().reorder(yi,rd, x, yo);
    argminfunc.compute_at(argminfunc2,x).update().reorder(di,rdi,yi).vectorize(di,native_lanes);
    blur_y.compute_at(argminfunc2,x).update().reorder(di,d,ry).vectorize(di,native_lanes);
    vsum.compute_at(argminfunc2,x).store_at(argminfunc2, yo).vectorize(di,native_lanes).fold_storage(x,2);
    f1.compute_at(argminfunc2,x).vectorize(di, 8);
    zero_blur.compute_at(argminfunc2, yo).vectorize(di, 8);

    return output;

}



Func diff_blur_bad(ImageParam input0, int height, int winsize, int depth, int tilesize){
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


    diff(di,yi,xi,yo,xo)=abs(b0(yi+yo*tilesize,xi+xo*tilesize)-b0(yi+yo*tilesize+di,xi+xo*tilesize));

    RDom rx(1, tilesize-1, "rx");

    RDom rk(0,winsize,"rk");
    Func vsum("vsum");
    vsum(di,yi,xi,yo,xo) = undef<uint>();
    vsum(di,0,xi,yo,xo) = sum(diff(di,rk,xi,yo,xo));
    vsum(di,rx,xi,yo,xo)=vsum(di,rx-1,xi,yo,xo)+diff(di,rx+winsize-1,xi,yo,xo)-diff(di,rx-1,xi,yo,xo);

    g(di,yi,xi,yo,xo) = undef<uint>();
    g(di,yi,0,yo,xo) = sum(vsum(di,yi,rk,yo,xo));
    g(di,yi,rx,yo,xo) = g(di,yi,rx-1,yo,xo)+ vsum(di,yi,rx+tilesize-1,yo,xo)-vsum(di,yi,rx-1,yo,xo);

    Func preout("preout");





    preout(di,y,x)=g(di,y%tilesize,x%tilesize,y/tilesize,x/tilesize);

    //g.compute_root();

    

    

    RDom rd(0,depth,"rd");
    Func argminfunc;
    output(y,x)=argmin(preout(rd,y,x), argminfunc)[0];
    

    RVar ri("ri"), ro("ro");
    argminfunc.update().split(rd, ro, ri, native_lanes);
    Var dii("dii");
    Func intermediate = argminfunc.update().rfactor({{ri, dii}});
    intermediate.reorder(dii, y, x, ro).vectorize(dii, native_lanes);
    //output(y,x)=sum(preout(rd,y,x));


    vsum.compute_at(g,yo).reorder(di,yi,xi,yo,xo).vectorize(di,native_lanes).update().reorder(di,xi,yo,xo).vectorize(di,native_lanes);
    vsum.update(1).reorder(di,rx,xi,yo,xo).vectorize(di,native_lanes);

    g.vectorize(yi,native_lanes).update().vectorize(di,native_lanes);
    g.update(1).reorder(yi,rx,yo,xo).vectorize(di,native_lanes);

    output.tile(y,x,yo,xo,yi,xi,tilesize,tilesize);

    g.compute_at(intermediate,ro);

    return output;
}


bool diff_blur_s1(){
    const int row = 1024;
    const int acc = 1024;
    const int depth = 16; 
    const int winsize = 20;
    const int tilesize = 64;

    ImageParam A(Int(32), 2, "input");

    Func result = diff_blur_bad(A, 1024, winsize, depth, tilesize);

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
    argmin_rfactor_test();
    printf("Running normal IIR...\n");
    //iir_normal();
    //box_blur_s2();
    printf("Running inductive IIR...\n");
    //iir_inductive();
    diff_blur_s1();
    printf("Running normal box blur...\n");
    //box_blur_normal();
    printf("Running normal sum scan...\n");
    //sum_normal();
    printf("Running inductive sum scan...\n");
    //sum_inductive();
    return 0;
}