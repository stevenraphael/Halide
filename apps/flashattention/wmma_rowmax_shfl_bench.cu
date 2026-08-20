// Throughput benchmark for: tensor-core tile multiply (mma.sync) + per-row max
// via warp shuffles, NO shared memory. Many independent warps, each doing one
// 16x8 = 16x16 * 16x8 tile and reducing its rows. Timed with CUDA events.

#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>

#define CHECK(x) do{ cudaError_t e=(x); if(e){printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} }while(0)

static __device__ __forceinline__ unsigned pack(__half lo, __half hi) {
    __half2 h = __halves2half2(lo, hi); unsigned u; memcpy(&u,&h,4); return u;
}

// Each warp does REPS back-to-back mma+rowmax to amplify compute over the
// global loads (which stay L1-resident: all warps read the same A,B).
template<int REPS>
__global__ void bench(const __half *Amem, const __half *Bmem, float *out, int nwarp) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= nwarp) return;
    int lane=threadIdx.x&31, group=lane>>2, tid=lane&3;

    unsigned a0=pack(Amem[group*16+tid*2],   Amem[group*16+tid*2+1]);
    unsigned a1=pack(Amem[(group+8)*16+tid*2],Amem[(group+8)*16+tid*2+1]);
    unsigned a2=pack(Amem[group*16+tid*2+8], Amem[group*16+tid*2+9]);
    unsigned a3=pack(Amem[(group+8)*16+tid*2+8],Amem[(group+8)*16+tid*2+9]);
    unsigned b0=pack(Bmem[tid*2+group*16],   Bmem[tid*2+1+group*16]);
    unsigned b1=pack(Bmem[tid*2+8+group*16], Bmem[tid*2+9+group*16]);

    float acc = 0.f;
    #pragma unroll 1
    for (int r=0; r<REPS; r++) {
        float c0=0.f,c1=0.f,c2=0.f,c3=0.f;
        // perturb B per-rep so nothing is optimized away
        unsigned bb0 = b0 ^ (r & 1), bb1 = b1;
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(bb0),"r"(bb1));
        float mt=fmaxf(c0,c1), mb=fmaxf(c2,c3);
        for (int off=1; off<=2; off<<=1) {
            mt=fmaxf(mt,__shfl_xor_sync(0xffffffffu,mt,off));
            mb=fmaxf(mb,__shfl_xor_sync(0xffffffffu,mb,off));
        }
        acc += mt + mb;
    }
    if (tid==0) out[warp] = acc;
}

int main() {
    const int M=16,K=16,N=8;
    const int REPS=256;
    const int nwarp = 1<<20;          // ~1M tiles per launch (x REPS)
    __half hA[M*K], hB[K*N];
    for (int i=0;i<M*K;i++) hA[i]=__float2half(sinf(0.3f*i));
    for (int i=0;i<K*N;i++) hB[i]=__float2half(cosf(0.17f*i));

    __half *dA,*dB; float *dO;
    CHECK(cudaMalloc(&dA,sizeof hA)); CHECK(cudaMalloc(&dB,sizeof hB));
    CHECK(cudaMalloc(&dO,nwarp*sizeof(float)));
    CHECK(cudaMemcpy(dA,hA,sizeof hA,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB,hB,sizeof hB,cudaMemcpyHostToDevice));

    int block=256, grid=(nwarp*32 + block-1)/block;
    // warmup
    for(int i=0;i<3;i++) bench<REPS><<<grid,block>>>(dA,dB,dO,nwarp);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0,ev1; cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    const int iters=50;
    cudaEventRecord(ev0);
    for(int i=0;i<iters;i++) bench<REPS><<<grid,block>>>(dA,dB,dO,nwarp);
    cudaEventRecord(ev1); CHECK(cudaEventSynchronize(ev1));
    float ms; cudaEventElapsedTime(&ms,ev0,ev1); ms/=iters;

    double tiles = (double)nwarp * REPS;
    double flops = tiles * 2.0 * M * N * K;       // matmul flops
    printf("tiles/launch=%.3g  REPS=%d\n", (double)nwarp, REPS);
    printf("time = %.4f ms   %.2f TFLOP/s (tensor-core matmul)   %.3g tiles/s\n",
           ms, flops/(ms*1e9), tiles/(ms*1e-3));
    cudaFree(dA);cudaFree(dB);cudaFree(dO);
    return 0;
}
