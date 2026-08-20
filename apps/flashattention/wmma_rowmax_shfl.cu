// Multiply two tiles with tensor cores (mma.sync) and compute the max of each
// row of the product -- WITHOUT any shared memory. The row reduction is done
// entirely with warp shuffles (__shfl_xor_sync), exploiting the documented
// register layout of the m16n8k16 mma accumulator. This is what real
// FlashAttention does for the softmax row statistics, and is exactly what
// Halide's opaque nvcuda::wmma path cannot express.
//
//   D[16x8] = A[16x16] * B[16x16->col-major, N=8]      (one warp, tensor core)
//   rowmax[m] = max_n D[m][n]                          (shfl_xor, no shared mem)
//
// Build/run (sm_89):
//   nvcc -arch=sm_89 wmma_rowmax_shfl.cu -o wmma_rowmax_shfl && ./wmma_rowmax_shfl

#include <cstdio>
#include <cmath>
#include <cuda_fp16.h>

#define CHECK(x) do{ cudaError_t e=(x); if(e){printf("CUDA err %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1;} }while(0)

static __device__ __forceinline__ unsigned pack(__half lo, __half hi) {
    __half2 h = __halves2half2(lo, hi);
    unsigned u; memcpy(&u, &h, 4); return u;
}

// One warp. A: 16x16 row-major (Amem[row*16+col]).
// B: 16x8, col-major (Bmem[k + n*16] = B[k][n]).  D = A*B, rowmax over n.
__global__ void wmma_rowmax(const __half *Amem, const __half *Bmem, float *rowmax) {
    int lane  = threadIdx.x & 31;
    int group = lane >> 2;      // 0..7
    int tid   = lane & 3;       // 0..3

    // --- load A fragment (8 halves/thread), matching the m16n8k16 .row layout
    unsigned a0 = pack(Amem[group*16 + tid*2+0],       Amem[group*16 + tid*2+1]);
    unsigned a1 = pack(Amem[(group+8)*16 + tid*2+0],   Amem[(group+8)*16 + tid*2+1]);
    unsigned a2 = pack(Amem[group*16 + tid*2+8],       Amem[group*16 + tid*2+9]);
    unsigned a3 = pack(Amem[(group+8)*16 + tid*2+8],   Amem[(group+8)*16 + tid*2+9]);

    // --- load B fragment (4 halves/thread), .col layout
    unsigned b0 = pack(Bmem[(tid*2+0) + group*16], Bmem[(tid*2+1) + group*16]);
    unsigned b1 = pack(Bmem[(tid*2+8) + group*16], Bmem[(tid*2+9) + group*16]);

    // --- tensor-core multiply: D = A*B, accumulator in registers c0..c3
    float c0=0.f, c1=0.f, c2=0.f, c3=0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));

    // Accumulator layout (m16n8k16):
    //   c0 = D[group  ][tid*2+0]   c1 = D[group  ][tid*2+1]
    //   c2 = D[group+8][tid*2+0]   c3 = D[group+8][tid*2+1]
    // So row `group` (cols tid*2+0/1) and row `group+8` are each spread over the
    // 4 lanes sharing `group`. Reduce over those 4 lanes with shfl_xor.
    float m_top = fmaxf(c0, c1);   // row = group
    float m_bot = fmaxf(c2, c3);   // row = group+8
    for (int off = 1; off <= 2; off <<= 1) {
        m_top = fmaxf(m_top, __shfl_xor_sync(0xffffffffu, m_top, off));
        m_bot = fmaxf(m_bot, __shfl_xor_sync(0xffffffffu, m_bot, off));
    }
    // All 4 lanes in the group now hold the row max; lane with tid==0 writes it.
    if (tid == 0) {
        rowmax[group]     = m_top;
        rowmax[group + 8] = m_bot;
    }
}

int main() {
    const int M=16, K=16, N=8;
    __half hA[M*K], hB[K*N];
    float A[M*K], B[K*N];
    // deterministic pseudo-random inputs
    for (int i=0;i<M*K;i++){ float v = sinf(0.3f*i)*1.0f; A[i]=v; hA[i]=__float2half(v); }
    for (int i=0;i<K*N;i++){ float v = cosf(0.17f*i)*1.0f; B[i]=v; hB[i]=__float2half(v); }

    __half *dA,*dB; float *dR;
    CHECK(cudaMalloc(&dA,sizeof hA)); CHECK(cudaMalloc(&dB,sizeof hB)); CHECK(cudaMalloc(&dR,M*sizeof(float)));
    CHECK(cudaMemcpy(dA,hA,sizeof hA,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB,hB,sizeof hB,cudaMemcpyHostToDevice));

    wmma_rowmax<<<1,32>>>(dA,dB,dR);
    CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());

    float R[M]; CHECK(cudaMemcpy(R,dR,M*sizeof(float),cudaMemcpyDeviceToHost));

    // reference: D[m][n] = sum_k A[m][k]*B[k + n*16]; rowmax[m]=max_n D[m][n]
    int ok=1;
    for (int m=0;m<M;m++){
        float best=-1e30f;
        for (int n=0;n<N;n++){ float d=0; for(int k=0;k<K;k++) d+=A[m*K+k]*B[k + n*16]; best=fmaxf(best,d); }
        float rel = fabsf(R[m]-best)/(fabsf(best)+1e-6f);
        printf("row %2d: gpu=%9.4f  ref=%9.4f  %s\n", m, R[m], best, rel<1e-2f?"ok":"MISMATCH");
        if (rel>=1e-2f) ok=0;
    }
    printf("%s\n", ok?"PASS (tensor-core matmul + warp-shuffle rowmax, NO shared memory)":"FAIL");
    cudaFree(dA);cudaFree(dB);cudaFree(dR);
    return ok?0:1;
}
