"""The TRUE SOTA reference: the production mamba_ssm fused CUDA selective_scan
kernel (hand-written, shared-memory chunked). Benchmarked at the same problem
sizes as the Halide GPU kernels. Our problem = 1 sequence, dim=D channels,
state=N, length=T (batch=1). Semantics differ only in discretization (Mamba
uses deltaA=exp(delta*A)); the operation, cost, and parallel structure are the
same diagonal selective scan, so the timing is the honest comparison."""

import sys
import time

import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
D = int(sys.argv[2]) if len(sys.argv) > 2 else 512
T = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
dev = "cuda"
torch.manual_seed(5)

batch = 1
u = torch.randn(batch, D, T, device=dev)
delta = torch.rand(batch, D, T, device=dev)  # positive
A = -torch.rand(D, N, device=dev)  # negative -> stable
B = torch.randn(batch, N, T, device=dev)
C = torch.randn(batch, N, T, device=dev)


def fused():
    return selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_softplus=False)


def bench(fn):
    fn()
    torch.cuda.synchronize()
    best = 1e18
    for _ in range(5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


t_fused = bench(fused)
# Correctness: fused vs the package's own pure-torch reference.
yf = fused()
yr = selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_softplus=False)
rel = float((yf - yr).abs().max() / (yr.abs().max() + 1e-6))
print(
    f"Mamba SOTA fused CUDA kernel ({torch.cuda.get_device_name(0)})  N={N} D={D} T={T} (batch=1)"
)
print(
    f"  mamba_ssm selective_scan_fn (fused CUDA): {t_fused:8.3f} ms   (rel vs ref {rel:.1e})"
)
