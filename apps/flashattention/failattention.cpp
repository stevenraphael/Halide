// Tensor-core (HMMA) attention on CUDA via the Halide Tile-IR backend.
//
// This is the NON-flash form: it materializes the N x N score matrix, but it
// runs BOTH matmuls (Q.K^T and P.V) on tensor cores through the CUDATileIR
// backend (the same LowerMma path proven by test/correctness/tile_ir_matmul).
// Softmax runs as ordinary CUDA reduction kernels in between.
//
// Layout rule that makes LowerMma fold an HMMA (no transpose_vector shuffle):
// read each matmul's contraction axis as the inner / stride-1 axis.
//   S(kj, qi) = sum_d Q(d,qi) * K(d,kj)      -> contraction d is inner for Q,K
//   O(d,  qi) = sum_kj P(kj,qi) * Vt(kj,d)   -> contraction kj is inner for P,Vt
// so V is supplied transposed as Vt(kj, d).
//
// Build:
//   g++ attention_mma.cpp -O2 -I <build>/include -I <src>/runtime -std=c++17 \
//       <build>/src/libHalide.so -lpthread -ldl -o attention_mma
//   HL_TILEIRAS=<cuda>/bin/tileiras \
//   LD_LIBRARY_PATH=<build>/src:/usr/lib/wsl/lib ./attention_mma

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace Halide;

static void naive_attention(const std::vector<float> &Q,
                            const std::vector<float> &Kk,
                            const std::vector<float> &V,
                            std::vector<float> &out, int N, int D) {
    std::vector<float> s(N);
    float scale = 1.f / std::sqrt((float)D);
    for (int i = 0; i < N; i++) {
        float mx = -1e30f;
        for (int j = 0; j < N; j++) {
            float dot = 0;
            for (int k = 0; k < D; k++) dot += Q[k + i * D] * Kk[k + j * D];
            s[j] = dot * scale;
            mx = std::max(mx, s[j]);
        }
        float lsum = 0;
        for (int j = 0; j < N; j++) { s[j] = std::exp(s[j] - mx); lsum += s[j]; }
        for (int dd = 0; dd < D; dd++) {
            float v = 0;
            for (int j = 0; j < N; j++) v += s[j] * V[dd + j * D];
            out[dd + i * D] = v / lsum;
        }
    }
}

static void set_ab(OutputImageParam p, int d0, int d1) {
    p.set_host_alignment(16).dim(0).set_bounds(0, d0).set_stride(1)
        .dim(1).set_bounds(0, d1).set_stride(d0);
}

int main() {
    const int N = 64, D = 4;
    const float scale = 1.f / std::sqrt((float)D);

    Var qi("qi"), kj("kj"), d("d");

    // Q,K stored (d, n): d is dim0/stride-1 (contraction inner for S).
    ImageParam Q(Float(16), 2, "Q");
    ImageParam Kp(Float(16), 2, "K");

    // ---- GEMM 1: S(kj, qi) = sum_d Q(d,qi) * K(d,kj) ----
    // Minimal repro: ONLY the score GEMM. Everything after S removed. The
    // multi-vectorized atomic reduction below produces the Q*K MultiRamp Mul
    // that CSE mishandles (IR.cpp:254 predicate-lanes assert) -- no tile-IR,
    // no softmax, no second GEMM needed.
    RDom rd(0, D, "rd");
    Func S("S");
    S(kj, qi) += cast<float>(Q(rd, qi)) * cast<float>(Kp(rd, kj));

    // ---- schedule ----
    Var xi("xi"), yi("yi");
    RVar ko("ko"), ki("ki");

    Func Sout = S.in();
    Sout.compute_root().tile(kj, qi, xi, yi, 16, 16);//
    Sout.bound(kj, 0, N).bound(qi, 0, N);
    S.compute_at(Sout, kj).vectorize(kj).vectorize(qi)
        .update().atomic().split(rd, ko, ki, D)
        .vectorize(kj).vectorize(qi).reorder(ki, kj, qi, ko).vectorize(ki);

    set_ab(Q, D, N);
    set_ab(Kp, D, N);
    set_ab(Sout.output_buffer(), N, N);
    S.bound(kj, 0, N).bound(qi, 0, N);

    Target target = Target("host-cuda");

    // ---- data ----
    Buffer<float16_t> Qb(D, N), Kb(D, N);
    Qb.fill(float16_t(1.f)); Kb.fill(float16_t(1.f));
    Qb.set_host_dirty(); Kb.set_host_dirty();
    Q.set(Qb); Kp.set(Kb);

    Buffer<float> C(N, N);

    try {
        Sout.compile_jit(target);
        Sout.realize(C, target);
        C.copy_to_host();
    } catch (const Halide::Error &e) {
        fprintf(stderr, "Halide error: %s\n", e.what());
        return 2;
    }

    printf("S GEMM compiled + ran. S(0,0)=%.1f (expect %d)\n", C(0, 0), D);
    return 0;
}
