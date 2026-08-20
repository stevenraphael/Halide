// Two-stage (time-parallel) GPU mamba selective scan -- the fair GPU counterpart
// to the vendor v1 kernel, which parallelizes the time axis. The mamba SSM step
// is a first-order LINEAR recurrence  h <- a*h + b  (a=exp(dt*A), b=dt*B*u), an
// AFFINE map, and affine maps compose associatively:
//     (a2,b2) o (a1,b1) = (a2*a1, a2*b1 + b2).
// So we scan time with the SAME two-stage chunk decomposition prefixsum_gpu uses,
// but composing (a,b) pairs instead of summing scalars. t = k*L + j (T = C*L):
//   ctotA(k) = prod_{r<L} a_r,   ctotB(k) = fold_{r<L}(a_r*.+b_r)   per-chunk map (parallel over k)
//   carryB(k)= h entering chunk k = exclusive compose of previous chunk maps       (serial O(C))
//   h(j,k)   = a_j*h(j-1,k)+b_j  starting from carryB(k)  -> inductive over j, folded
//   y(t)     = sum_n C_n*h_n(t) + D*u(t)
//
// Layout: channel d is dim0 (contiguous) so warps coalesce. Parallel axes:
// chunk k, batch b, channel d (tiled to threads); state n unrolled (N<=16);
// j and the carry-over-k are the only serial parts. Constant B/C, D term,
// delta_softplus -- matching mamba_v1_bench so the ms/Mtok/s are comparable.
#include "../support/bench_harness.h"
#include "Halide.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
using namespace Halide;

int main(int argc, char **argv) {
    const int D = argc > 1 ? atoi(argv[1]) : 2048;
    const int N = argc > 2 ? atoi(argv[2]) : 16;
    int T = argc > 3 ? atoi(argv[3]) : 8192;
    const int B = argc > 4 ? atoi(argv[4]) : 8;
    int L = argc > 5 ? atoi(argv[5]) : 256;   // chunk length (serial fold depth); sweep this
    if (T % L != 0) { T = (T / L) * L; if (T == 0) { fprintf(stderr, "T>=L\n"); return 1; } }
    const int C = T / L;                      // chunks (parallel over GPU blocks)

    try {
        Target target = get_host_target().with_feature(Target::CUDA).with_feature(Target::CUDACapability86);

        // Inputs, channel-major (dim0 = d) for coalesced loads.
        Buffer<float> ub(D, T, B), deltab(D, T, B), Ab(D, N), Bb(D, N), Cb(D, N), Db(D);
        srand(1);
        auto rnd = [] { return (float)rand() / RAND_MAX; };
        for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) for (int d = 0; d < D; d++) {
            ub(d, t, b) = rnd() - 0.5f; deltab(d, t, b) = rnd() * 0.5f; }
        for (int n = 0; n < N; n++) for (int d = 0; d < D; d++) {
            Ab(d, n) = -(0.1f + rnd()); Bb(d, n) = rnd() - 0.5f; Cb(d, n) = rnd() - 0.5f; }
        for (int d = 0; d < D; d++) Db(d) = rnd();

        Var d("d"), n("n"), j("j"), k("k"), b("b");

        // Per-step affine coefficients (inline; recomputed on the up- and down-sweep,
        // like prefixsum_gpu reads its input twice).
        Func dt("dt"), a("a"), bc("bc");
        dt(d, j, k, b) = log(1.0f + exp(deltab(d, k * L + j, b)));   // softplus
        a(d, n, j, k, b)  = exp(dt(d, j, k, b) * Ab(d, n));
        bc(d, n, j, k, b) = dt(d, j, k, b) * Bb(d, n) * ub(d, k * L + j, b);

        // Stage 1 (up-sweep): per-chunk affine map, parallel over (d,n,k,b).
        // Tuple {A_prod, B_acc}: A_prod = prod a_r ; B_acc = fold (a_r*.+b_r) from 0.
        // Fusing the two into ONE reduction means a_r=exp(dt*A) is computed once per
        // (d,n,r) (CSE dedups within the single update stmt), halving the up-sweep
        // transcendentals vs two separate funcs.  Serial over r.
        RDom r(0, L, "r");
        Func ctot("ctot");
        ctot(d, n, k, b) = Tuple(1.0f, 0.0f);
        Expr ar = a(d, n, r, k, b), br = bc(d, n, r, k, b);
        ctot(d, n, k, b) = Tuple(ar * ctot(d, n, k, b)[0],
                                 ar * ctot(d, n, k, b)[1] + br);
        Func ctotA = ctot;  // {0}=A_prod, {1}=B_acc

        // Stage 2 (carry): h entering chunk k = exclusive compose of prior chunk maps.
        // carryB_k = ctotA_{k-1}*carryB_{k-1} + ctotB_{k-1}.  Serial O(C) over k.
        Func carryB(Float(32), "carryB");
        carryB(d, n, k, b) = select(k <= 0, 0.0f,
            likely(ctot(d, n, k - 1, b)[0] * carryB(d, n, k - 1, b) + ctot(d, n, k - 1, b)[1]));

        // Stage 3 (down-sweep): local scan seeded with the carry, inductive over j, folded.
        Func h(Float(32), "h");
        h(d, n, j, k, b) = select(j <= 0,
            a(d, n, 0, k, b) * carryB(d, n, k, b) + bc(d, n, 0, k, b),
            likely(a(d, n, j, k, b) * h(d, n, j - 1, k, b) + bc(d, n, j, k, b)));

        // Output: y_t = sum_n C_n h_n + D u.
        RDom rn(0, N, "rn");
        Func y("y");
        y(d, j, k, b) = Db(d) * ub(d, k * L + j, b);
        y(d, j, k, b) += Cb(d, rn) * h(d, rn, j, k, b);

        // ---- GPU schedule: d tiled to threads, (do,k,b) blocks; n serial ----
        Var di("di"), doo("doo");
        const int TH = std::min(128, D);
        // dt (softplus: log+exp) is n-independent but otherwise recomputed for every
        // n on BOTH sweeps (2N times/element). Materialize once: blocks (do,j,k,b)? no
        // -- keep it simple: threads on d, blocks over (do,k,b), j serial inside.
        dt.compute_root().bound(j, 0, L).bound(k, 0, C).bound(b, 0, B)
          .reorder(j, d, k, b).gpu_tile(d, doo, di, TH).gpu_blocks(k, b);
        // Up-sweep per-chunk maps: blocks over (do,k,b), threads di; r,n serial
        // INNERMOST (inside the thread) so no loop sits between GPU block loops.
        ctot.compute_root().reorder(n, d, k, b).gpu_tile(d, doo, di, TH).gpu_blocks(k, b);
        ctot.update(0).reorder(r, n, d, k, b).gpu_tile(d, doo, di, TH).gpu_blocks(k, b);
        // Carry: parallel over (do,b) + threads di; k serial (inductive) + n inside thread.
        carryB.compute_root().reorder(n, k, d, b).gpu_tile(d, doo, di, TH).gpu_blocks(b);
        // Down-sweep: blocks (do,k,b), threads di; j serial (the fold), n serial.
        y.bound(d, 0, D).bound(j, 0, L).bound(k, 0, C).bound(b, 0, B)
         .reorder(j, d, k, b).gpu_tile(d, doo, di, TH).gpu_blocks(k, b);
        y.update(0).reorder(rn, j, d, k, b).gpu_tile(d, doo, di, TH).gpu_blocks(k, b).unroll(rn);
        h.compute_at(y, j).store_at(y, di).fold_storage(j, hb::fold_factor(1, L));

        y.compile_jit(target);

        Buffer<float> result(D, L, C, B);
        y.realize(result);
        result.copy_to_host();
        hb::Stats st = hb::bench([&] { y.realize(result); result.device_sync(); });
        result.copy_to_host();

        // Correctness vs serial CPU reference.
        double err = -1.0; bool ok = true;
        if (getenv("CHECK")) {
            err = 0.0; std::vector<float> hh(N);
            for (int bb = 0; bb < B; bb++) for (int dd = 0; dd < D; dd++) {
                for (int nn = 0; nn < N; nn++) hh[nn] = 0.f;
                for (int t = 0; t < T; t++) {
                    float dtv = std::log(1.f + std::exp(deltab(dd, t, bb)));
                    float uv = ub(dd, t, bb), yv = Db(dd) * uv;
                    for (int nn = 0; nn < N; nn++) {
                        hh[nn] = std::exp(dtv * Ab(dd, nn)) * hh[nn] + dtv * Bb(dd, nn) * uv;
                        yv += Cb(dd, nn) * hh[nn];
                    }
                    float got = result(dd, t % L, t / L, bb);
                    err = std::max(err, (double)std::fabs(got - yv) / (std::fabs(yv) + 1e-4));
                }
            }
            ok = err < 2e-2;
        }

        const double mtok = (double)B * T / 1e6;
        const double state_bytes = (double)D * N * C * B * 3 * 4;  // ctotA/B + carryB
        char note[192];
        snprintf(note, sizeof(note),
                 "mamba 2-stage (time-parallel scan)  B=%d D=%d N=%d T=%d L=%d C=%d", B, D, N, T, L, C);
        hb::print_spec_header("mamba_gpu_2stage", target.to_string(), note);
        hb::print_row("mamba Halide 2-stage (time-parallel)", st, mtok / (st.min * 1e-3), "Mtok/s",
                      state_bytes, err, ok);
        printf("  %s\n", (getenv("CHECK") && !ok) ? "FAILED" : "Success!");
        return (getenv("CHECK") && !ok) ? 1 : 0;
    } catch (const Halide::Error &e) {
        fprintf(stderr, "EXCEPTION: %s\n", e.what());
        return 1;
    }
}
