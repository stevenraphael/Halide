"""SOTA third-party baseline for the Mamba GPU head-to-head, via the production
Mamba-2 SSD kernel `mamba_chunk_scan_combined` (hand-written, Triton-JIT, chunked
shared-memory scan). Used instead of selective_scan_fn because the prebuilt
selective_scan CUDA extension in this env is ABI-incompatible with the installed
torch (segfaults in selective_scan_fwd); the Triton SSD path compiles at runtime.

Matched to our problem: 1 sequence, D channels, state N, length T (batch=1). We
map D->nheads with headdim=1, ngroups=1, dstate=N, so the kernel performs the same
D*N*T diagonal selective-scan work. As noted in the repo's other baselines, the
discretization differs (SSD uses deltaA=exp(dt*A)); the operation, cost, and
parallel structure are the same diagonal scan, so the TIMING is the honest
comparison. Same methodology as mamba_gpu_bench: warmup then median of 30 trials.
"""

import sys, time
import torch
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
D = int(sys.argv[2]) if len(sys.argv) > 2 else 512
T = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
# headdim: SSD is tuned for 64-128. headdim=1 maps D->nheads 1:1 (same layout as
# our Halide kernel) but is a degenerate regime for the SSD kernel. To compare
# fairly we keep the TOTAL scan work identical (nheads*headdim*dstate*T = D*N*T)
# while giving the kernel a realistic headdim: nheads = D/headdim, dstate = N.
headdim = int(sys.argv[4]) if len(sys.argv) > 4 else 64
batch = (
    int(sys.argv[5]) if len(sys.argv) > 5 else 16
)  # match the batched inductive test
dev = "cuda"
dtype = torch.float32
torch.manual_seed(5)

if D % headdim != 0:
    raise SystemExit(f"D={D} must be divisible by headdim={headdim}")
nheads, ngroups, dstate = D // headdim, 1, N
chunk_size = 256

x = torch.randn(batch, T, nheads, headdim, device=dev, dtype=dtype)
dt = torch.rand(batch, T, nheads, device=dev, dtype=dtype)  # positive
A = -torch.rand(nheads, device=dev, dtype=dtype)  # stable
B = torch.randn(batch, T, ngroups, dstate, device=dev, dtype=dtype)
C = torch.randn(batch, T, ngroups, dstate, device=dev, dtype=dtype)


def fused():
    return mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None)


def bench(fn, warmup=3, trials=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(trials):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2], ts[len(ts) // 4], ts[3 * len(ts) // 4]


med, p25, p75 = bench(fused)
mtoks = batch * D * T / (med * 1e3)  # match inductive: batch*D*T tokens
name = torch.cuda.get_device_name(0)
print(
    f"Mamba-2 SSD (Triton) fused kernel  N={N} D={D} T={T} batch={batch}  "
    f"(nheads={nheads} headdim={headdim} dstate={dstate})  ({name})"
)
print(
    f"  {'mamba_chunk_scan_combined':30s}  {med:10.3f} {p25:10.3f} {p75:10.3f}   {mtoks:10.1f}  (median/p25/p75 ms, Mtok/s)"
)
