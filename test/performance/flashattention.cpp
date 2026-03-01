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

Func flashattention(ImageParam &Q, ImageParam &K, ImageParam &V, int idim, int jdim, int i2dim, int kdim, int tilesize) {
    Var x("x"), y("y"), xi("xi"), yi("yi"), xo("xo"), yo("yo"), d("d"), di("di");
    RDom rmul(0, kdim);
    Func S("S");
    S(xi, yi, xo, yo, d) = Halide::cast<float>(Expr((float)0.0));
    S(xi, yi, xo, yo, d) += Q(xi+xo*tilesize, rmul, d) * K(xi+xo*tilesize, rmul, d);
    Func rowmaxintm("rowmaxintm");
    RDom rowvar(0, tilesize);
    rowmaxintm(yi, xo, yo, d) = Halide::cast<float>(Expr((float)-INFINITY));
    rowmaxintm(yi, xo, yo, d) = max(rowmaxintm(yi, xo, yo, d), S(rowvar, yi, xo, yo, d));
    Func rowsumintm("rowsumintm");
    rowsumintm(yi, xo, yo, d) = Halide::cast<float>(Expr((float)0.0));
    rowsumintm(yi, xo, yo, d) += exp(S(rowvar, yi, xo, yo, d) - rowmaxintm(yi, xo, yo, d));
    Func P1intm("P1intm");
    P1intm(xi, yi, xo, yo, d) = Halide::cast<float>(Expr((float)0.0));
    P1intm(xi, yi, xo, yo, d) += exp(S(rowvar, yi, xo, yo, d) - rowmaxintm(yi, xo, yo, d)) * V(xi+xo*tilesize, rowvar+yo*tilesize, d) / rowsumintm(yi, xo, yo, d);
    Func rowmax("rowmax");
    rowmax(yi, xo, yo, d) = select(xo < 0, Halide::cast<float>(Expr((float)-INFINITY)), max(rowmax(yi, xo-1, yo, d), rowmaxintm(yi, xo, yo, d)));
    Func rowsum("rowsum");
    rowsum(yi, xo, yo, d) = select(xo < 0, Halide::cast<float>(Expr((float)0.0)), exp(rowmaxintm(yi,xo,yo,d)-rowmax(yi, xo, yo, d))*rowsumintm(yi, xo, yo, d)+exp(rowmax(yi, xo-1, yo, d)-rowmax(yi, xo, yo, d))*rowsum(yi, xo-1, yo, d));
    Func P1("P1");
    Expr new_p1_1 = exp(rowmax(yi,xo-1,yo,d)-rowmax(yi, xo, yo, d))*rowsum(yi, xo-1, yo, d)*P1(xi,yi,xo-1,yo,d);
    Expr new_p1_2 = exp(rowmaxintm(yi,xo,yo,d)-rowmax(yi, xo, yo, d))*rowsumintm(yi, xo, yo, d)*P1intm(xi, yi, xo, yo, d);
    Expr combined = (new_p1_1 + new_p1_2) / rowsum(yi, xo, yo, d);
    P1(xi, yi, xo, yo, d) = select(xo < 0, Halide::cast<float>(Expr((float)0.0)), combined);


    Func output("output");
    output(x,y,d) = P1(x, y%tilesize, idim/tilesize, y/tilesize, d);

    P1.compute_root().gpu_blocks(yo).gpu_threads(xi,yi).fold_storage(xo,2);
    rowmax.compute_at(P1, xo).store_at(P1, yo).fold_storage(xo, 2);
    rowsum.compute_at(P1, xo).store_at(P1, yo).fold_storage(xo, 2);

    Q.in(S).compute_at(P1, xo);//.copy_to_device();
    K.in(S).compute_at(P1, xo);
    V.in(P1intm).compute_at(P1, xo);
    
    S.compute_at(P1, xo).gpu_threads(xi, yi).update().gpu_threads(xi, yi);
    rowmaxintm.compute_at(P1, xo).gpu_threads(yi).update().gpu_threads(yi);
    rowsumintm.compute_at(P1, xo).gpu_threads(yi).update().gpu_threads(yi);
    P1intm.compute_at(P1, xo).gpu_threads(xi, yi).update().gpu_threads(xi, yi);

    return output;
    
    


}

Func flash2(int idim, int jdim, int i2dim, int kdim, int tilesize) {
    Var x("x"), y("y"), xi("xi"), yi("yi"), xo("xo"), yo("yo"), d("d"), di("di");
    RDom rmul(0, kdim);
    Func Q("Q"), K("K"), V("V");
    Q(xi, yi, d) = Halide::cast<float>(Expr((float)0.0));
    K(xi, yi, d) = Halide::cast<float>(Expr((float)0.0));
    V(xi, yi, d) = Halide::cast<float>(Expr((float)0.0));
    Func S("S");
    S(xi, yi, xo, yo, d) = Halide::cast<float>(Expr((float)0.0));
    S(xi, yi, xo, yo, d) += Q(xi+xo*tilesize, rmul, d) * K(xi+xo*tilesize, rmul, d);
    Func rowmaxintm("rowmaxintm");
    RDom rowvar(0, tilesize);
    rowmaxintm(yi, xo, yo, d) = Halide::cast<float>(Expr((float)-INFINITY));
    rowmaxintm(yi, xo, yo, d) = max(rowmaxintm(yi, xo, yo, d), S(rowvar, yi, xo, yo, d));
    Func rowsumintm("rowsumintm");
    rowsumintm(yi, xo, yo, d) = Halide::cast<float>(Expr((float)0.0));
    rowsumintm(yi, xo, yo, d) += exp(S(rowvar, yi, xo, yo, d) - rowmaxintm(yi, xo, yo, d));
    Func P1intm("P1intm");
    P1intm(xi, yi, xo, yo, d) = Halide::cast<float>(Expr((float)0.0));
    P1intm(xi, yi, xo, yo, d) += exp(S(rowvar, yi, xo, yo, d) - rowmaxintm(yi, xo, yo, d)) * V(xi+xo*tilesize, rowvar+yo*tilesize, d) / rowsumintm(yi, xo, yo, d);
    Func rowmax("rowmax");
    rowmax(yi, xo, yo, d) = select(xo < 0, Halide::cast<float>(Expr((float)-INFINITY)), max(rowmax(yi, xo-1, yo, d), rowmaxintm(yi, xo, yo, d)));
    Func rowsum("rowsum");
    rowsum(yi, xo, yo, d) = select(xo < 0, Halide::cast<float>(Expr((float)0.0)), exp(rowmaxintm(yi,xo,yo,d)-rowmax(yi, xo, yo, d))*rowsumintm(yi, xo, yo, d)+exp(rowmax(yi, xo-1, yo, d)-rowmax(yi, xo, yo, d))*rowsum(yi, xo-1, yo, d));
    Func P1("P1");
    Expr new_p1_1 = exp(rowmax(yi,xo-1,yo,d)-rowmax(yi, xo, yo, d))*rowsum(yi, xo-1, yo, d)*P1(xi,yi,xo-1,yo,d);
    Expr new_p1_2 = exp(rowmaxintm(yi,xo,yo,d)-rowmax(yi, xo, yo, d))*rowsumintm(yi, xo, yo, d)*P1intm(xi, yi, xo, yo, d);
    Expr combined = (new_p1_1 + new_p1_2) / rowsum(yi, xo, yo, d);
    P1(xi, yi, xo, yo, d) = select(xo < 0, Halide::cast<float>(Expr((float)0.0)), combined);


    Func output("output");
    output(x,y,d) = P1(x, y%tilesize, idim/tilesize, y/tilesize, d);

    P1.compute_root().gpu_blocks(yo).gpu_threads(xi,yi).fold_storage(xo,2);
    rowmax.compute_at(P1, xo).store_at(P1, yo).fold_storage(xo, 2);
    rowsum.compute_at(P1, xo).store_at(P1, yo).fold_storage(xo, 2);

    Q.compute_root().in(S).compute_at(P1, xo).copy_to_device();
    K.compute_root().in(S).compute_at(P1, xo).copy_to_device();
   
    
    S.compute_at(P1, xo).gpu_threads(xi, yi).update().gpu_threads(xi, yi);
    rowmaxintm.compute_at(P1, xo).gpu_threads(yi).update().gpu_threads(yi);
    rowsumintm.compute_at(P1, xo).gpu_threads(yi).update().gpu_threads(yi);
    P1intm.compute_at(P1, xo).gpu_threads(xi, yi).update().gpu_threads(xi, yi);

    return output;

}

int flashattention_test(){
    int idim = 1024;
    int jdim = 64;
    int i2dim = 1024;
    int kdim = 4;
    int tilesize = 32;

    ImageParam Q(Float(32), 3, "Q");
    ImageParam K(Float(32), 3, "K");
    ImageParam V(Float(32), 3, "V");

    Func f = flashattention(Q, K, V, idim, jdim, i2dim, jdim, tilesize);

    Buffer<float> q_buf(idim, jdim, kdim);
    Buffer<float> k_buf(idim, jdim, kdim);
    Buffer<float> v_buf(jdim, i2dim, kdim);
    // fill q_buf, k_buf, and v_buf with some data here if desired
        for (int i = 0; i < idim; i++) {
        for (int j = 0; j < jdim; j++) {
            for (int d = 0; d < kdim; d++) {
                q_buf(i, j, d) = static_cast<float>(i + j + d);
                k_buf(i, j, d) = static_cast<float>(i - j + d);
            }
        }
    }
    for (int i = 0; i < i2dim; i++) {
        for (int j = 0; j < jdim; j++) {
            for (int d = 0; d < kdim; d++) {
                v_buf(i, j, d) = static_cast<float>(i * j * d);
            }
        }
    }
    
    Q.set(q_buf);
    K.set(k_buf);
    V.set(v_buf);

    
    Buffer<float> output(jdim, i2dim, kdim);
    Target target = get_jit_target_from_environment().with_feature(Target::CUDA);
    /*auto time = Tools::benchmark(5, 5, [&]() {
        f.realize(output, target);
    });*/
    std::cout << "Exec time: " << time << "\n";
    f.compile_to_conceptual_stmt("flashattention_stmt.txt", {Q,K,V}, Text, target);

    return 0;
}


int main(int argc, char **argv) {
    Target target = get_jit_target_from_environment();
    //argmin_rfactor_test();
    printf("Running flashattention test...\n");
    //iir_normal();
    //box_blur_s2();
    //iir_inductive();

    //bounds_test_1();
    //hist_test();
    //diff_blur_1d_good();
    //diff_blur_1d_bad();


    flashattention_test();
    
    return 0;
}