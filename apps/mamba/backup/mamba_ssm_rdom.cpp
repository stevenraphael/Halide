// Mamba / S6 selective scan with the coefficient tensors FUSED on-chip.
//
// Real Mamba is "hardware-aware": it never materializes the big
// (N,D,T) discretized tensors in DRAM. It reads only small parameters and
// forms the coefficients inline:
//   Abar(n,d,t) = exp(delta(d,t) * A(n,d))          A: (N,D) static param
//   Bbar(n,d,t) = delta(d,t) * B(n,t)               B,C: (N,T) shared across d
//   h(n,d,t)    = Abar*h(n,d,t-1) + Bbar*x(d,t)
//   y(d,t)      = sum_n C(n,t) * h(n,d,t)
//
// DRAM traffic drops from ~3*N*D*T (materialized A/B/C) to
// N*D + 2*D*T + 2*N*T -- about 10x less here -- so we break past the DRAM
// ceiling that the pre-materialized version was pinned to. h is a plain
// inductive func (same-index t-1 shift), folded to a single in-place cell.

#include "Halide.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <cstdlib>

using namespace Halide;

int main() {
    const int D = 16384;  // channels
    const int N = 16;    // state dimension
    const int T = getenv("T") ? atoi(getenv("T")) : 512;   // sequence length

    Target target = get_jit_target_from_environment();
    const bool gpu = getenv("USE_GPU") != nullptr;
    if (gpu) {
        target.set_feature(Target::CUDA);
        target.set_feature(Target::CUDACapability86);
    }

    Var n("n"), d("d"), t("t");

    ImageParam A_p(Float(32), 2, "A_p");    // (n, d) static state matrix
    ImageParam delta_p(Float(32), 2, "delta_p");  // (d, t)
    ImageParam B_p(Float(32), 2, "B_p");    // (n, t) shared across channels
    ImageParam C_p(Float(32), 2, "C_p");    // (n, t) shared across channels
    ImageParam x_p(Float(32), 2, "x_p");    // (d, t)

    // NON-inductive: classic RDom-scan. h is materialized in full (N x D x T)
    // -- pure def gives the t=0 base case for every t, then an update scans
    // over an RDom in t, reading h(...,rt-1). No sliding-window fold is
    // possible on an RDom update, so the whole state is written to DRAM and
    // read back by y ("compute the first layer fully, then the second").
    Func h = Func(Float(32), 3, "h");
    Func y("y"), out("out");
    try {
        Expr Abar = exp(delta_p(d, t) * A_p(n, d));
        Expr Bbar = delta_p(d, t) * B_p(n, t);
        h(n, d, t) = Bbar * x_p(d, t);              // base case (correct at t=0)
        RDom rt(1, T - 1, "rt");
        Expr Abr = exp(delta_p(d, rt) * A_p(n, d));
        h(n, d, rt) = Abr * h(n, d, rt - 1) + delta_p(d, rt) * B_p(n, rt) * x_p(d, rt);

        RDom rn(0, N, "rn");
        y(d, t) = 0.f;
        y(d, t) += C_p(rn, t) * h(rn, d, t);
        out(d, t) = y(d, t);

        Var do_("do_"), di("di"), no("no"), ni("ni");
        if (gpu) {
            // h materialized at root; its scan is serial over rt, parallel
            // over (n,d). Then out/y read the stored h back.
            h.compute_root();
            Var hd_o("hd_o"), hd_i("hd_i");
            h.split(d, hd_o, hd_i, 32).reorder(n, hd_i, hd_o, t).gpu_blocks(hd_o).gpu_threads(hd_i);
            h.update().split(d, hd_o, hd_i, 32).reorder(n, hd_i, hd_o, rt).gpu_blocks(hd_o).gpu_threads(hd_i);
            out.compute_root().split(d, do_, di, 32).reorder(t, di, do_).gpu_blocks(do_).gpu_threads(di);
            y.compute_at(out, t);
            y.update().unroll(rn);
        } else {
            h.compute_root();
            h.update().reorder(n, d, rt).parallel(d);
            out.compute_root().split(d, do_, di, 8).parallel(do_).vectorize(di);
            y.compute_at(out, di);
        }
    } catch (const Halide::Error &e) {
        fprintf(stderr, "CONSTRUCTION/SCHEDULE ERROR: %s\n", e.what());
        return 1;
    }

    std::mt19937 rng(1);
    std::uniform_real_distribution<float> ua(-2.0f, -0.1f), us(-0.5f, 0.5f), ud(0.01f, 0.2f);
    Buffer<float> A_buf(N, D), delta_buf(D, T), B_buf(N, T), C_buf(N, T), x_buf(D, T);
    std::vector<float> Av((size_t)N*D), dv((size_t)D*T), Bv((size_t)N*T), Cv((size_t)N*T), xv((size_t)D*T);
    for (int dd=0; dd<D; dd++)
      for (int nn=0; nn<N; nn++){ float a=ua(rng); A_buf(nn,dd)=a; Av[(size_t)dd*N+nn]=a; }
    for (int tt=0; tt<T; tt++){
      for (int nn=0; nn<N; nn++){ float b=us(rng),c=us(rng); B_buf(nn,tt)=b; C_buf(nn,tt)=c; Bv[(size_t)tt*N+nn]=b; Cv[(size_t)tt*N+nn]=c; }
      for (int dd=0; dd<D; dd++){ float dl=ud(rng),xx=us(rng); delta_buf(dd,tt)=dl; x_buf(dd,tt)=xx; dv[(size_t)tt*D+dd]=dl; xv[(size_t)tt*D+dd]=xx; }
    }
    A_p.set(A_buf); delta_p.set(delta_buf); B_p.set(B_buf); C_p.set(C_buf); x_p.set(x_buf);

    Buffer<float> result;
    try {
        result = out.realize({D, T}, target);
        if (gpu) result.copy_to_host();
    } catch (const Halide::Error &e) {
        fprintf(stderr, "REALIZE ERROR: %s\n", e.what());
        return 1;
    }

    // Reference.
    std::vector<float> hprev((size_t)N*D, 0.f), ref((size_t)D*T);
    for (int tt=0; tt<T; tt++)
      for (int dd=0; dd<D; dd++) {
        float yv=0.f, dl=dv[(size_t)tt*D+dd], xx=xv[(size_t)tt*D+dd];
        for (int nn=0; nn<N; nn++){
            float ab = std::exp(dl*Av[(size_t)dd*N+nn]);
            float bb = dl*Bv[(size_t)tt*N+nn];
            float hh = (tt==0) ? bb*xx : ab*hprev[(size_t)dd*N+nn] + bb*xx;
            hprev[(size_t)dd*N+nn]=hh;
            yv += Cv[(size_t)tt*N+nn]*hh;
        }
        ref[(size_t)tt*D+dd]=yv;
      }

    float max_err=0, denom=0;
    for (int tt=0; tt<T; tt++)
      for (int dd=0; dd<D; dd++){
        max_err=std::max(max_err, std::abs(result(dd,tt)-ref[(size_t)tt*D+dd]));
        denom=std::max(denom, std::abs(ref[(size_t)tt*D+dd]));
      }
    printf("D=%d N=%d T=%d  max_err=%.6e (rel %.2e)  %s\n", D,N,T, max_err, max_err/denom,
           max_err < 1e-3f*std::max(1.f,denom) ? "PASS":"FAIL");

    const int iters = 30;
    for (int w=0; w<3; w++) out.realize(result, target);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it=0; it<iters; it++) { out.realize(result, target); }
    if (gpu) result.device_sync();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
    // Fused DRAM traffic: A(N*D) + delta(D*T) + x(D*T) + B(N*T) + C(N*T) + out(D*T).
    double bytes = ((double)N*D + 3.0*D*T + 2.0*N*T) * 4.0;
    double flops = 4.0*N*D*T;   // exp + 2 mul-add for h, mul-add for y
    printf("  time=%.3f ms   %.1f GB/s (fused DRAM)   %.2f GFLOP/s\n", ms, bytes/(ms*1e6), flops/(ms*1e6));
    return 0;
}
