# Benchmark suite: algorithms, implementations, and swept parameters

## Common methodology

Every benchmark is a first-order recurrence whose per-step state can be *folded*
(stored in a small sliding window) rather than *materialized* over the whole
recurrence axis. To isolate the effect of storage folding from the effect of
producer/consumer fusion, each app is expressed as **three variants that share
the same algorithm and fusion structure and differ only in how much of the
recurrence state is kept live:**

1. **non-inductive (materialize)** — the baseline formulation. The recurrence is
   written *without* inductive functions: the full trajectory (or a dense sliding
   window via an `RDom`) is materialized in a buffer. This is the "obvious"
   Halide implementation.
2. **inductive UNFOLDED** — the inductive-function formulation with storage
   folding *disabled* (fold factor pinned to the full recurrence-axis extent).
   Same fusion as the folded version, but the whole trajectory is still live.
   This variant isolates *folding* from *fusion*: it differs from (3) only in the
   fold factor.
3. **inductive FOLDED** — the inductive-function formulation with the recurrence
   axis folded to the minimal window (typically 2–3 slots). This is the proposed
   implementation.

Where a mature third-party or hand-written baseline exists (OpenCV for stereo,
Boost.odeint for the ODE, oneTBB/CUB/Thrust for scan, the vendor CUDA kernel for
Mamba) it is included **as an external reference**, but it is not a
"non-inductive" formulation in the sense above and is reported separately.

**Dimension roles.** For each app we distinguish three kinds of axis, because
folding interacts with each differently:

- **recurrence length** — the axis the recurrence runs along, i.e. the *folded*
  axis. Sweeping it grows the unfolded footprint without changing per-step work;
  it is the axis that crosses the cache boundary.
- **per-step work** — the amount of computation per recurrence step (transition
  scan width, matvec size, stencil width). Sweeping it changes arithmetic
  intensity.
- **batch / parallel** — independent problem instances scheduled in parallel.
  Sweeping it grows total footprint via a *parallel* axis, so its points collapse
  onto the same footprint/cache curve as the recurrence-length sweep.

**Measurement protocol.** Timing: median of 30 timed iterations after 3 warmups
(best/min and IQR also recorded); each iteration is one full `realize()` (plus a
device sync on GPU). Runs are pinned to a single NUMA node. Memory: the
recurrence-state footprint is reported **byte-exact** — either measured as the
high-water mark of a custom Halide allocator wrapped around one untimed
`realize()` (viterbi, chebyshev, ode) or computed analytically from the schedule
(kalman, prefixsum, stereobm). Input/output buffers are excluded so only the
recurrence state — the quantity folding controls — is counted. Footprint is
deterministic (independent of CPU speed and, for these apps, of thread count),
so it is captured with a single trial (`run_mem.sh`).

CPU results were taken on an AMD EPYC (Zen 2, AVX2) node; GPU results on an
NVIDIA A10G (sm_86). All Halide variants use the same Halide build.

---

## CPU benchmarks

### 1. Viterbi decoding (`viterbi_log`)

**Algorithm.** Maximum-likelihood state-sequence decoding for a hidden Markov
model in the log domain. At each time step the per-state score is updated by a
max-plus transition scan over the previous step's scores plus the emission term;
a backpointer is retained for the traceback. The forward score pass is a
first-order recurrence over time `t`.

**Implementations.** non-inductive (materialize the full `S × T` score/backpointer
trajectory) · inductive UNFOLDED (`fold t → T+1`) · inductive FOLDED
(`fold t → 2`, only the current and previous columns live).

**Parameters swept.**
- (A) **recurrence length** `T ∈ {5000, 20000, 80000, 320000}` at `S=16` states,
  `M=4`, single-thread (serial recurrence, no parallel axis).
- (B) **per-step work** `S ∈ {4, 16, 64, 256}` states at `M=8`, `T=50000`
  (grows the O(S) transition scan and the footprint together).

### 2. Kalman / latent AR(2) log-likelihood (`kalman_ar`)

**Algorithm.** Log-likelihood of a latent second-order autoregressive process
with observation noise, evaluated by the Kalman filter recursion (predict/update)
over time. The filtered mean/covariance form a first-order recurrence over `t`;
the per-instance log-likelihood accumulates along the way. Batched over `B`
independent series.

**Implementations.** non-inductive (materialize the full filtered trajectory) ·
inductive UNFOLDED (`fold t → T`) · inductive FOLDED (`fold t → 2`).

**Parameters swept.**
- (A) **recurrence length** `T ∈ {1024, 4096, 16384, 65536, 131072}` at `B=256`,
  at both `HL_NUM_THREADS=1` and `32` (serial vs parallel-over-batch).
- (B) **batch (parallel axis)** `B ∈ {16, 64, 256, 1024}` at `T=16384`, threads
  capped to the task count.
- (C) **arithmetic intensity** at `B=256, T=16384`: the per-step divide/log
  replaced by a fixed-reciprocal multiply (`HB_CHEAP`), same footprint — tests
  whether a folding loss is latency-bound rather than footprint-bound.

### 3. Chebyshev semi-iteration (`chebyshev_inductive`)

**Algorithm.** Chebyshev semi-iterative solver for a linear system: each
iteration is a dense matrix-vector product plus a two-term recurrence combining
the current and two previous iterates (the Chebyshev three-term recurrence). The
recurrence length is the iteration count `M` (~convergence count); per-step work
is the O(n²) matvec.

**Implementations.** non-inductive FULL materialize (`M+1` iterate columns) ·
non-inductive mod-3 ring (3 columns, hand-written modulo indexing) · inductive
UNFOLDED (`fold → M+1 cols`) · inductive FOLDED (`fold → 3 cols`).

**Parameters swept.**
- (A) **per-step work** `n ∈ {512, 1024, 2048, 3072}` at `M=100` iterations
  (grows the O(n²) matvec).
- (B) **recurrence length / fold ratio** `M ∈ {50, 100, 200, 400}` at `n=2048`.

This solver's footprint (`n·M·8`) stays cache-resident at realistic sizes, so it
is the intended *in-cache* datapoint (fold ≈ tie), contrasting the out-of-cache
sequence apps.

### 4. Allen–Cahn ODE integration (`ode_observer_sparse_fused_test`)

**Algorithm.** Method-of-lines integration of the Allen–Cahn reaction–diffusion
PDE with a two-step Adams–Bashforth (AB2) scheme. The spatial field of width `D`
is advanced in time by a sparse stencil; AB2 makes the update a first-order
recurrence needing the two previous derivative evaluations. A fused observer
reduces the trajectory on the fly. Batched over `B`.

**Implementations.** non-inductive (materialize the full space×time trajectory) ·
inductive UNFOLDED (`fold t → T+1`) · inductive FOLDED (`fold t → 3`,
AB2 window). External references: Boost.odeint (RK integrator) and a C++
reference + observer oracle (for correctness).

**Parameters swept.**
- (A) **recurrence length** `T ∈ {2048, 8192, 32768, 131072}` at `D=1024`, `B=1`,
  single-thread.
- (B) **batch** `B ∈ {1, 4, 16}` at `D=1024`, `T=8192`.

### 5. Batched prefix scan (`prefixsum_bench` and variants)

**Algorithm.** Inclusive prefix sum over each row of a `W × H` matrix, followed
by a running-mean (or cheap `>>2`) consumer. The scan is a first-order additive
recurrence along `W`. Because `+` is associative it also admits a two-stage
chunk decomposition that parallelizes *along* the scan axis.

**Implementations.** inductive FOLDED (`fold x → 1`, O(1) accumulator) ·
inductive UNFOLDED (`fold x → W`, materialized prefix row) · non-inductive
(`RDom`, materialize the O(W) row) · one-stage FOLDED (parallel over independent
rows only) · two-stage inductive scan (parallel along time via chunking).
External reference: oneTBB `parallel_scan` (and, on GPU, CUB/Thrust — reported
separately as vendor scans, not non-inductive formulations).

**Parameters swept.**
- **Full matrix** `W=1048576, H=32`: cross of consumer ∈ {running mean, `>>2`}
  × threads ∈ {1, capped}, comparing fold / unfold / non-inductive RDom / oneTBB.
- **Parallelism regime**: lane count `S ∈ {1, 8, 32}` at `W=1048576`, comparing
  one-stage (row-parallel only) vs oneTBB vs two-stage (time-parallel, chunk
  `L=4096`) — shows the two-stage recovering parallelism at small lane counts.

### 6. Stereo block matching (`stereobm_jit`)

**Algorithm.** StereoBM disparity estimation: for each pixel and candidate
disparity, a sum-of-absolute-differences over a `winsize × winsize` support
window; the disparity minimizing SAD is selected. The vertical accumulation of
the SAD box filter is a first-order recurrence down image rows `y` (a sliding
window), which is what folds.

**Implementations.** inductive FOLDED (`fold y → 1`, single running row) ·
inductive UNFOLDED (`fold y → H`, full column) · schedule4 (non-inductive,
`RDom` vertical slide). External reference: OpenCV StereoBM on the same image
pair (timing baseline; its memory is outside Halide's allocator).

**Parameters swept.** Real image pairs (full-res aloe `1282×1110`, and the small
aloe `307×265`), `winsize=15` (SAD window, fixed per config), single-thread, with
the strip width **tilesize ∈ {8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 128}**
swept densely. Tilesize is a pure scheduling knob (it does not change the result,
so OpenCV parity holds); it maps how strip width trades cache locality against
the size of the materialized non-inductive/unfolded state.

---

## GPU benchmark (reported separately)

### Mamba / S6 selective scan

**Algorithm.** The Mamba selective-scan step is a first-order **affine** (linear)
recurrence per (batch, channel, state): `h[t] = a[t]·h[t-1] + b[t]`, with
`a=exp(dt·A)`, `b=dt·B·u`, `dt=softplus(delta)`; the output is a pointwise
reduction `y[t] = D·u[t] + Σ_n C[n]·h[n,t]`. Only `h` is a true recurrence.

**Implementations.** Halide register-fold inductive scan (serial-in-time, `h`
kept in registers) · Halide two-stage time-parallel scan (affine-map composition
`(a₂,b₂)∘(a₁,b₁)=(a₂a₁, a₂b₁+b₂)` over chunks) · vendor CUDA v1 kernel
(`selective_scan`, official state-spaces implementation). **There is no
non-inductive Mamba baseline** — materializing all of `h` is ≈8 GiB and exceeds
the buffer size limit — so Mamba is presented as an inductive-Halide-vs-vendor
comparison, not an inductive-vs-non-inductive one.

**Parameters swept** (realistic Mamba-1 dims `D=2048, N=16`): recurrence length
`T ∈ {4096, 8192, 16384, 65536}` at `B=8`; batch/occupancy `B ∈ {1, 4, 8, 16}`
at `T=8192`; and, for the two-stage kernel, chunk length `L` (fold depth /
time-parallelism trade-off).
