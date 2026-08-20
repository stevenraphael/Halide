// Ablated FlashAttention, SIMPLEST form: ONE tile of QK^T, then a per-row max.
// No inductive funcs, no tiling. One warp per query i; its 32 lanes each own
// one key-position L. The row max is a plain reduction across the 32 lanes
// (regular RDom -> warp shuffle), kept per-lane in registers.
//
//   score(L,i) = sum_d Q(d,i)*K(d,L)          (QK^T -- the FLOPs, one per lane)
//   rowmax(u,i)= max_L score(L,i)             (regular reduction over lanes,
//                                              replicated to every lane u -> reg)
//   out(i)     = rowmax(0,i)
#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>
using namespace Halide;
int main(){
    Target target=get_jit_target_from_environment();
    target.set_feature(Target::CUDA); target.set_feature(Target::CUDACapability86);
    const int D=128, W=32;
    const int Ni = getenv("NI")?atoi(getenv("NI")):8192;
    const int Nj = W;   // a SINGLE tile of keys
    Var L("L"), u("u"), i("i"), io("io"), ii("ii");
    ImageParam Q(Float(32),2,"Q"), K(Float(32),2,"K");
    Func score("score"), rowmax("rowmax"), out("out");
    Buffer<float> outb;
    try {
        RDom rd(0,D,"rd");
        score(L,i) = 0.f;
        score(L,i) += Q(rd,i)*K(rd, L);

        // NO redundant per-lane "u": rowmax(i) is one warp-scalar, reduced over
        // the 32 lanes (rL) by warp shuffle. score(rL) is striped across lanes.
        RDom rL(0,W,"rL");
        rowmax(i) = -std::numeric_limits<float>::infinity();
        rowmax(i) = max(rowmax(i), score(rL,i));

        out(i) = rowmax(i);

        // ---- schedule: 32 queries per warp, query i on the lanes. score and
        // rowmax are computed INSIDE the lane loop -> lane-local registers,
        // each lane owns one query's full 32-key row, reduces it serially. ----
        out.split(i,io,ii,W).gpu_blocks(io).gpu_lanes(ii);
        rowmax.compute_at(out,ii).store_in(MemoryType::Register);
        score.compute_at(out,ii).store_in(MemoryType::Register);

        std::mt19937 rng(0); std::uniform_real_distribution<float> uu(-1,1);
        Buffer<float> Qb(D,Ni), Kb(D,Nj);
        for(int a=0;a<Ni;a++)for(int e=0;e<D;e++)Qb(e,a)=uu(rng);
        for(int a=0;a<Nj;a++)for(int e=0;e<D;e++)Kb(e,a)=uu(rng);
        Q.set(Qb); K.set(Kb);
        outb=out.realize({Ni}, target); outb.copy_to_host();
        for(int a=0;a<12;a++){ float best=-1e30f;
          for(int jj=0;jj<Nj;jj++){ float s=0; for(int e=0;e<D;e++) s+=Qb(e,a)*Kb(e,jj); best=std::max(best,s);}
          printf("row %d: halide=%.4f ref=%.4f\n", a, outb(a), best);
        }
    } catch(const Halide::Error&e){ fprintf(stderr,"ERR: %s\n",e.what()); return 1; }
    const int it=50; for(int w=0;w<3;w++) out.realize(outb,target); outb.device_sync();
    auto t0=std::chrono::high_resolution_clock::now();
    for(int k=0;k<it;k++) out.realize(outb,target);
    outb.device_sync();
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/it;
    printf("Ni=%d Nj=%d D=%d  %.4f ms  %.2f TFLOP/s\n", Ni,Nj,D, ms, (2.0*Ni*Nj*D)/(ms*1e9));
    return 0;
}
