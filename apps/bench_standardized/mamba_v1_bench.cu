// Torch-free standalone benchmark for the OFFICIAL mamba v1 selective-scan CUDA
// kernel (state-spaces/mamba, csrc/selective_scan) -- the vendor GPU baseline for
// our Halide inductive selective scan (mamba_ssm USE_GPU=1 / mamba_gpu_batched).
//
// It includes the real kernel header and instantiates selective_scan_fwd_cuda<
// float,float> directly, driving it with cudaMalloc'd raw pointers (no PyTorch).
// The four c10 headers it needs are shimmed in mamba_v1_shims/ (type-only).
//
// Layout matches selective_scan.cpp: u,delta,out = [B,D,T] contiguous;
// A,B,C = [D,N] (constant B/C, n_groups=1); x workspace = [B,D,n_chunks,2N];
// n_chunks = ceil(T/2048); delta_softplus = true; has_z = false; D term present.
//
// Build (on the measurement machine, CUDA 13 + A10G sm_86):
//   nvcc -O3 -std=c++17 -arch=sm_86 --expt-relaxed-constexpr \
//     -I $HOME/mamba_src/csrc/selective_scan \
//     -I apps/bench_standardized/mamba_v1_shims -I apps/support \
//     apps/bench_standardized/mamba_v1_bench.cu -o mamba_v1_bench
// Run:   ./mamba_v1_bench [D N T B]      (CHECK=1 validates vs a CPU reference)

#include <cstdint>       // uint64_t used by SSMParamsBase before any torch header
#include <algorithm>     // std::max(initializer_list) used in selective_scan_common.h
#include <initializer_list>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "c10/util/Half.h"      // at::Half used by selective_scan_common.h Converter<>
#include "c10/util/BFloat16.h"  // at::BFloat16, same reason

#include "selective_scan.h"
#include "selective_scan_fwd_kernel.cuh"  // defines selective_scan_fwd_cuda<>
#include "bench_harness.h"

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 3;} } while(0)

int main(int argc, char **argv) {
    const int D = argc > 1 ? atoi(argv[1]) : 2048;   // channels (dim)
    const int N = argc > 2 ? atoi(argv[2]) : 16;     // state dim (dstate)
    const int T = argc > 3 ? atoi(argv[3]) : 8192;   // sequence length
    const int B = argc > 4 ? atoi(argv[4]) : 8;      // batch
    const bool delta_softplus = true;
    const int n_chunks = (T + 2048 - 1) / 2048;

    const size_t uN = (size_t)B * D * T, wN = (size_t)D * N, xN = (size_t)B * D * n_chunks * 2 * N;

    // Host init (small values so the scan stays finite; A negative for stability).
    std::vector<float> hu(uN), hdelta(uN), hA(wN), hB(wN), hC(wN), hD(D), hout(uN);
    srand(1);
    auto rnd = [](){ return (float)rand() / RAND_MAX; };
    for (auto &v : hu) v = rnd() - 0.5f;
    for (auto &v : hdelta) v = rnd() * 0.5f;          // pre-softplus
    for (auto &v : hA) v = -(0.1f + rnd());           // negative -> stable
    for (auto &v : hB) v = rnd() - 0.5f;
    for (auto &v : hC) v = rnd() - 0.5f;
    for (auto &v : hD) v = rnd();

    float *u, *delta, *A, *Bp, *Cp, *Dp, *out, *x;
    CK(cudaMalloc(&u, uN*4));    CK(cudaMalloc(&delta, uN*4));
    CK(cudaMalloc(&A, wN*4));    CK(cudaMalloc(&Bp, wN*4)); CK(cudaMalloc(&Cp, wN*4));
    CK(cudaMalloc(&Dp, D*4));    CK(cudaMalloc(&out, uN*4)); CK(cudaMalloc(&x, xN*4));
    CK(cudaMemcpy(u, hu.data(), uN*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(delta, hdelta.data(), uN*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(A, hA.data(), wN*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Bp, hB.data(), wN*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Cp, hC.data(), wN*4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Dp, hD.data(), D*4, cudaMemcpyHostToDevice));
    CK(cudaMemset(x, 0, xN*4));

    SSMParamsBase p;
    memset(&p, 0, sizeof(p));
    p.batch = B; p.dim = D; p.seqlen = T; p.dstate = N; p.n_groups = 1;
    p.n_chunks = n_chunks; p.dim_ngroups_ratio = D; p.delta_softplus = delta_softplus;
    p.is_variable_B = false; p.is_variable_C = false;
    p.u_ptr = u; p.delta_ptr = delta; p.A_ptr = A; p.B_ptr = Bp; p.C_ptr = Cp;
    p.D_ptr = Dp; p.delta_bias_ptr = nullptr; p.out_ptr = out; p.x_ptr = x;
    p.z_ptr = nullptr; p.out_z_ptr = nullptr;
    // strides in ELEMENTS. Contiguous [B,D,T]: batch=D*T, d=T. [D,N]: d=N, dstate=1.
    p.A_d_stride = N; p.A_dstate_stride = 1;
    p.B_d_stride = N; p.B_dstate_stride = 1;
    p.C_d_stride = N; p.C_dstate_stride = 1;
    p.u_batch_stride = (size_t)D*T; p.u_d_stride = T;
    p.delta_batch_stride = (size_t)D*T; p.delta_d_stride = T;
    p.out_batch_stride = (size_t)D*T; p.out_d_stride = T;

    // Warm-up + validate launch succeeds.
    selective_scan_fwd_cuda<float, float>(p, 0);
    CK(cudaDeviceSynchronize());

    hb::Stats s = hb::bench([&]{ selective_scan_fwd_cuda<float, float>(p, 0); cudaDeviceSynchronize(); });

    double err = -1.0; bool ok = true;
    if (getenv("CHECK")) {
        CK(cudaMemcpy(hout.data(), out, uN*4, cudaMemcpyDeviceToHost));
        err = 0.0;
        std::vector<float> h(N);
        for (int b = 0; b < B; b++) for (int d = 0; d < D; d++) {
            for (int n = 0; n < N; n++) h[n] = 0.f;
            for (int t = 0; t < T; t++) {
                float dr = hdelta[((size_t)b*D+d)*T+t];
                float dt = delta_softplus ? log1pf(expf(dr)) : dr;   // softplus
                float ut = hu[((size_t)b*D+d)*T+t];
                float y = hD[d]*ut;
                for (int n = 0; n < N; n++) {
                    float dA = expf(dt*hA[d*N+n]);
                    float dBu = dt*hB[d*N+n]*ut;
                    h[n] = dA*h[n] + dBu;
                    y += hC[d*N+n]*h[n];
                }
                float got = hout[((size_t)b*D+d)*T+t];
                float den = fabsf(y) + 1e-4f;
                err = std::max(err, (double)(fabsf(got-y)/den));
            }
        }
        ok = err < 2e-2;   // fp32 kernel vs naive ref: loose tolerance
    }

    const double mtok = (double)B * T / 1e6;                 // sequences x length
    const double state_bytes = (double)xN * 4;
    char note[160];
    snprintf(note, sizeof(note),
             "mamba v1 selective_scan (official CUDA)  B=%d D=%d N=%d T=%d n_chunks=%d",
             B, D, N, T, n_chunks);
    hb::print_spec_header("mamba_v1_cuda", "cuda-sm_86", note);
    hb::print_row("mamba v1 selective_scan (vendor CUDA)", s, mtok/(s.min*1e-3), "Mtok/s",
                  state_bytes, err, ok);

    cudaFree(u); cudaFree(delta); cudaFree(A); cudaFree(Bp); cudaFree(Cp);
    cudaFree(Dp); cudaFree(out); cudaFree(x);
    return (getenv("CHECK") && !ok) ? 1 : 0;
}
