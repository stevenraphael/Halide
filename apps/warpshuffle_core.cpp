// Minimal warp-shuffle cross-lane reduction (the FlashAttention softmax-row
// reduction primitive), following Halide's register_shuffle rfactor idiom.
// Each warp reduces a length-32 dot product across its 32 lanes via shfl.sync
// -- no shared memory, no bar.sync. Batched over many warps for throughput.
#include "Halide.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
using namespace Halide;
int main(){
    Target target=get_jit_target_from_environment();
    target.set_feature(Target::CUDA); target.set_feature(Target::CUDACapability86);
    const int R=32;
    const int B = getenv("B")?atoi(getenv("B")):(1<<20);
    Var b("b"), u("u"), bo("bo"), bi("bi"), blane("blane");
    RVar ro("ro"), ri("ri");
    ImageParam Q(Float(32),2,"Q"), K(Float(32),2,"K");
    RDom r(0,R,"r");
    Func dot("dot"), outc("outc");
    Buffer<float> out;
    try {
        dot(b) = 0.f;
        dot(b) += Q(r,b)*K(r,b);
        outc(b) = dot(b);
        // Mirror register_shuffle's reduction idiom exactly: each group of
        // warp(=32) lanes reduces one output; batch across blocks/threads.
        const int warps_per_block = 4;
        outc.split(b, bo, bi, warps_per_block)
            .split(bi, bi, blane, 1)
            .gpu_blocks(bo).gpu_threads(bi, blane);
        Func intm = dot.update().split(r, ri, ro, R).reorder(ri, ro).rfactor(ro, u);
        intm.compute_at(outc, bi).update().gpu_lanes(u);
        intm.gpu_lanes(u);
        Buffer<float> Qb(R,B), Kb(R,B); Qb.fill(1.0f); Kb.fill(2.0f);
        Q.set(Qb); K.set(Kb);
        out=outc.realize({B}, target); out.copy_to_host();
    } catch(const Halide::Error&e){ fprintf(stderr,"ERR: %s\n",e.what()); return 1; }
    printf("check out(0)=%f (want %f)\n", out(0), (float)(R*2.0f));
    const int it=100; for(int w=0;w<3;w++) outc.realize(out,target); out.device_sync();
    auto t0=std::chrono::high_resolution_clock::now();
    for(int k=0;k<it;k++) outc.realize(out,target);
    out.device_sync();
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/it;
    printf("B=%d  %.4f ms  %.2f TFLOP/s\n", B, ms, (2.0*R*B)/(ms*1e9));
    return 0;
}
