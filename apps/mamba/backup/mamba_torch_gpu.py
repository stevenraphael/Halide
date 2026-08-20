"""Third-party GPU references for the Mamba selective-SSM scan (diagonal
recurrence h_t = a_t h_{t-1} + b_t x_t, y_t = sum_n c_t h_t), to sit next to the
Halide CUDA numbers. mamba_ssm's hand-written CUDA kernel is not installable
against torch 2.13+cu130, so we use the two standard fair PyTorch references:
  (1) SEQUENTIAL  = selective_scan_ref style: Python loop over t, vectorized over
      (D,N). This is the official reference implementation's algorithm.
  (2) CHUNKED PARALLEL = the same sqrt(T) two-stage algorithm as the Halide
      blocked kernel, expressed with torch cumprod/cumsum (fully parallel over t
      within a chunk; short sequential loop over the C chunk carries).
Timed with torch.cuda.synchronize, best-of-5, on the same (N,D,T)."""
import sys, time, math
import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
D = int(sys.argv[2]) if len(sys.argv) > 2 else 512
T = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
dev = "cuda"
torch.manual_seed(5)

# Match the C++ generator's ranges (values need not be bit-identical; the two
# torch methods validate against each other).
a = (0.9 + torch.randint(0, 99, (D, N, T), device=dev).float() / 1000.0)   # (D,N,T)
b = (torch.randint(0, 200, (N, T), device=dev).float() / 100.0 - 1.0)
c = (torch.randint(0, 200, (N, T), device=dev).float() / 100.0 - 1.0)
x = (torch.randint(0, 200, (D, T), device=dev).float() / 100.0 - 1.0)
B = b[None, :, :] * x[:, None, :]          # (D,N,T)  =  b_t x_t


def seq_ref(a, B, c):
    # Official-reference algorithm: sequential loop over time.
    D, Nn, Tn = a.shape
    y = torch.empty((D, Tn), device=dev)
    h = B[:, :, 0].clone()                  # h_0 = B_0
    y[:, 0] = (c[:, 0][None, :] * h).sum(1)
    for t in range(1, Tn):
        h = a[:, :, t] * h + B[:, :, t]
        y[:, t] = (c[:, t][None, :] * h).sum(1)
    return y


def chunked_parallel(a, B, c, L):
    # Two-stage sqrt(T) scan, vectorized. Intra-chunk scan via cumprod/cumsum;
    # inter-chunk carry via a short sequential loop over C chunks.
    D, Nn, Tn = a.shape
    C = (Tn + L - 1) // L
    pad = C * L - Tn
    ap = torch.ones((D, Nn, C * L), device=dev)
    Bp = torch.zeros((D, Nn, C * L), device=dev)
    ap[:, :, :Tn] = a
    Bp[:, :, :Tn] = B
    ap = ap.reshape(D, Nn, C, L)
    Bp = Bp.reshape(D, Nn, C, L)
    PA = torch.cumprod(ap, dim=-1)                       # prod_{i<=j} a
    PB = PA * torch.cumsum(Bp / PA, dim=-1)              # local h, carry-in 0
    cA = PA[:, :, :, -1]                                 # (D,N,C) chunk aggregates
    cB = PB[:, :, :, -1]
    carry = torch.zeros((D, Nn, C), device=dev)
    for k in range(1, C):                               # short serial carry scan
        carry[:, :, k] = cA[:, :, k - 1] * carry[:, :, k - 1] + cB[:, :, k - 1]
    h = PA * carry[:, :, :, None] + PB                  # (D,N,C,L)
    h = h.reshape(D, Nn, C * L)[:, :, :Tn]
    y = torch.einsum('nt,dnt->dt', c, h)
    return y


def bench(fn):
    fn(); torch.cuda.synchronize()
    best = 1e18
    for _ in range(5):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        r = fn(); torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best, r


L = int(math.ceil(math.sqrt(T)))
t_seq, y_seq = bench(lambda: seq_ref(a, B, c))
t_par, y_par = bench(lambda: chunked_parallel(a, B, c, L))
rel = float((y_seq - y_par).abs().max() / (y_seq.abs().max() + 1e-6))

print(f"Mamba GPU third-party (PyTorch, {torch.cuda.get_device_name(0)})  N={N} D={D} T={T}  L={L}")
print(f"  PyTorch sequential (selective_scan_ref style): {t_seq:8.3f} ms")
print(f"  PyTorch chunked-parallel (sqrt-T, torch ops):  {t_par:8.3f} ms   (rel seq-vs-par {rel:.1e})")
