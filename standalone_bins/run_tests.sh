#!/usr/bin/env bash
# Runs the Viterbi, Kalman (steady-state + local-linear-trend + AR(2)),
# stereo block-matching, prefix-sum, Chebyshev semi-iteration, Allen-Cahn ODE
# integration, and Mamba/S6 selective-scan benchmark executables in this
# folder across a variety of arguments, to sanity-check the standalone builds
# on a fresh machine.
#
# Everything here is self-contained: each executable finds libHalide.so.22 via
# its $ORIGIN rpath (a real file, not a symlink -- some transfer methods, e.g.
# browser upload, drop symlinks and exec permissions), so no Halide/LLVM
# install or build tools are needed on the target machine -- just run this
# script from wherever this folder ends up (e.g. after copying it to a fresh
# Ubuntu 24.04 box).
set -u
cd "$(dirname "$0")"
chmod +x -- *.sh viterbi viterbi_log kalman_ar \
    prefixsum_bench prefixsum_bench_rdom prefixsum_bench_tbb prefixsum_sweep \
    prefixsum_bench_onestage prefixsum_bench_fold mamba_gpu_batched \
    stereobm_process stereobm_process_opencv chebyshev_inductive \
    stereobm_process_w9_t64 stereobm_process_w9_t128 stereobm_process_w9_t32 \
    stereobm_process_opencv_w9_t64 stereobm_process_opencv_w9_t128 stereobm_process_opencv_w9_t32 \
    ode_observer_sparse_fused_test mamba_ssm \
    2>/dev/null
DATA_DIR="$(mktemp -d)"
trap 'rm -rf "$DATA_DIR"' EXIT

# The binaries here are compiled with -march=native (and stereobm's AOT with
# target=host) for the machine that BUILT them. For the paper's own numbers you
# must build on the measurement machine: run apps/bench_standardized/build_all.sh
# there, or invoke this script with REBUILD=1 (needs the source tree + distrib).
if [ "${REBUILD:-0}" = 1 ]; then
    BA="../apps/bench_standardized/build_all.sh"
    if [ -x "$BA" ]; then
        echo "REBUILD=1: rebuilding all binaries with -march=native for THIS machine..."
        "$BA" || {
            echo "!!! build_all.sh failed"
            exit 1
        }
    else
        echo "!!! REBUILD=1 set but $BA not found (run from a full source checkout)"
        exit 1
    fi
fi
# Preflight: a binary built for a different CPU dies with SIGILL (exit 132).
# Catch it up front with actionable guidance rather than a cryptic mid-run crash.
./viterbi_log 4 3 8 "$DATA_DIR/preflight.bin" >/dev/null 2>&1
if [ "$?" -eq 132 ]; then
    echo "!!! Illegal instruction (SIGILL) from a prebuilt binary."
    echo "!!! These were compiled with -march=native for a DIFFERENT CPU."
    echo "!!! Rebuild on THIS machine:  REBUILD=1 ./run_tests.sh"
    echo "!!!   (or run apps/bench_standardized/build_all.sh directly)"
    exit 1
fi

# NUMA pinning: bind every benchmark to ONE NUMA node (CPUs + memory) so a run's
# threads and its data live on the same socket -- kills the cross-socket variance
# that makes best_ms << median_ms on big multi-socket boxes. On by default when
# numactl exists; disable with PIN_NUMA=0, pick a node with NUMA_NODE=N.
PIN=""
if [ "${PIN_NUMA:-1}" = 1 ] && command -v numactl >/dev/null 2>&1; then
    NUMA_NODE="${NUMA_NODE:-0}"
    PIN="numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE"
    echo "# NUMA pinning ON: $PIN   (PIN_NUMA=0 to disable, NUMA_NODE=N to change node)"
    # Cores available on the pinned node cap the useful thread count.
    NODE_CPUS="$(numactl --hardware 2>/dev/null | sed -n "s/^node $NUMA_NODE cpus: //p" | wc -w)"
    [ -n "$NODE_CPUS" ] && [ "$NODE_CPUS" -gt 0 ] && NCORES_PIN="$NODE_CPUS"
elif [ "${PIN_NUMA:-1}" = 1 ]; then
    echo "# NUMA pinning requested but numactl not found -- running UNPINNED (expect noisier best_ms)"
fi
NCORES="$(nproc 2>/dev/null || echo 1)"
# When pinned to one node, the logical core ceiling is that node's cpu count.
NCORES="${NCORES_PIN:-$NCORES}"
# cap N: threads = min(nproc, N). Every app is parallel over a finite number of
# tasks (batch rows, tiles, chunks, lanes); asking for more threads than there
# are tasks just oversubscribes and drowns the signal in fork/join + allocator
# noise on a big box. So each run below caps HL_NUM_THREADS to its own task count.
# (viterbi/chebyshev/ode have NO .parallel() -> serial -> capped to 1.)
cap() {
    local n=$1
    [ "$n" -lt "$NCORES" ] && echo "$n" || echo "$NCORES"
}

pass=0
fail=0
run() {
    echo "=== $PIN $* ==="
    if $PIN "$@"; then
        pass=$((pass + 1))
    else
        echo "!!! FAILED: $* !!!"
        fail=$((fail + 1))
    fi
    echo
}

echo "############################################"
echo "# Viterbi (inductive decoder, no args)"
echo "############################################"
[ -x ./viterbi ] && run ./viterbi || echo "# (./viterbi not present -- skipped; see viterbi_log below)"

echo "############################################"
echo "# Viterbi bench (log domain): inductive fold vs non-inductive materialize"
echo "############################################"
# Dimension roles: S = per-step work (transition-scan width), T = recurrence
# length (folded axis). viterbi_log runs all 3 variants in one invocation.
# viterbi has NO parallel axis (single sequence, serial fold) -> 1 task -> 1 thread.
echo "### (A) recurrence-length sweep  HL_NUM_THREADS=1 (serial: no parallel axis)  S=16 fixed,  T crossing LLC ###"
for T in 5000 20000 80000 320000; do
    run env HL_NUM_THREADS=1 ./viterbi_log 16 4 "$T" "$DATA_DIR/viterbi_data.bin"
done
echo "### (B) state-count (per-step-work) sweep  T=50000 fixed,  S grows work+footprint ###"
for S in 4 16 64 256; do
    run ./viterbi_log "$S" 8 50000 "$DATA_DIR/viterbi_data.bin"
done

echo "############################################"
echo "# Kalman latent AR(2) log-likelihood (B T)"
echo "# Three labeled sweeps by DIMENSION ROLE (see paper methodology):"
echo "#  (A) recurrence-length T  : the folded axis; sweep across the LLC boundary."
echo "#  (B) batch B              : the parallel axis; sweep at FIXED T -- footprint"
echo "#                             B*NC*T*8 crosses LLC via B instead of T, so its"
echo "#                             points collapse onto the same fp/LLC curve as (A)."
echo "#  (C) arithmetic intensity : HB_CHEAP swaps the per-step divide/log for a"
echo "#                             fixed-reciprocal multiply (same footprint), which"
echo "#                             should flip folding from lose->win if the loss is"
echo "#                             latency-bound rather than footprint-bound."
echo "############################################"
# Parallel axis = batch, scheduled parallel(bo) with bo=B/8 -> tasks=B/8.
# (A) sweeps {1, min(nproc,B/8)}; at B=256 that's min(nproc,32).
for nt in 1 "$(cap $((256 / 8)))"; do
    echo "### (A) recurrence-length sweep  HL_NUM_THREADS=$nt (B/8=32 tasks)  B=256 fixed,  T crossing LLC ###"
    for T in 1024 4096 16384 65536; do
        run env HL_NUM_THREADS=$nt ./kalman_ar 256 "$T"
    done
done
echo "### (B) batch-size sweep  T=16384 fixed,  B crossing LLC (threads=min(nproc,B/8)) ###"
for B in 16 64 256 1024; do
    run env HL_NUM_THREADS=$(cap $((B / 8))) ./kalman_ar "$B" 16384
done
echo "### (C) arithmetic-intensity flip  B=256 T=16384  (divide/log vs cheap multiply, threads=min(nproc,32)) ###"
run env HL_NUM_THREADS=$(cap $((256 / 8))) ./kalman_ar 256 16384
run env HB_CHEAP=1 HL_NUM_THREADS=$(cap $((256 / 8))) ./kalman_ar 256 16384

echo "############################################"
echo "# StereoBM (JIT): our inductive fold-vs-unfold pipeline TIMED against OpenCV's"
echo "# StereoBM on the SAME real image. Image size is fixed by the pair (NOT a free"
echo "# axis); we run BOTH the small aloe pair (single-core) and the full-res pair"
echo "# (multicore). tilesize (strip width) is a pure scheduling knob -- it does not"
echo "# change the result, so OpenCV parity holds -- and is swept DENSELY (not just"
echo "# powers of 2) to map how strip width trades off cache locality vs parallelism."
echo "# winsize (SAD window / OpenCV parameter) fixed at 9."
echo "############################################"
# Parallel axis = the x-tiles, parallel(xo) with tiles=ceil(W/tilesize). Threads
# are capped to that tile count (small pair W=307, full-res W=1282), so a big box
# doesn't oversubscribe when few strips exist. The small pair is single-core anyway.
STEREOBM_TILESIZES="8 12 16 24 32 40 48 64 80 96 128"
ceil_div() { echo $((($1 + $2 - 1) / $2)); }
if [ -x ./stereobm_jit ]; then
    for TS in $STEREOBM_TILESIZES; do
        echo "### small aloe (307x265)  tilesize=$TS  winsize=9  threads=1 ###"
        run env HL_NUM_THREADS=1 STEREOBM_NUM_THREADS=1 WINSIZE=9 TILESIZE=$TS \
            ./stereobm_jit aloeL.png aloeR.png
    done
    if [ -f aloeL_large.png ] && [ -f aloeR_large.png ]; then
        for TS in $STEREOBM_TILESIZES; do
            nt=$(cap "$(ceil_div 1282 "$TS")") # tiles = ceil(W/tilesize), W=1282
            echo "### full-res aloe (1282x1110)  tilesize=$TS  winsize=9  threads=$nt (min(nproc, ceil(1282/$TS) tiles)) ###"
            run env HL_NUM_THREADS=$nt STEREOBM_NUM_THREADS=$nt WINSIZE=9 TILESIZE=$TS \
                ./stereobm_jit aloeL_large.png aloeR_large.png
        done
    else
        echo "# (aloeL_large.png / aloeR_large.png not present -- skipping full-res)"
    fi
fi

echo "############################################"
echo "# Prefix-sum + average: FULL MATRIX"
echo "#   variants  : fold (inductive, O(1) state) | materialize (non-inductive,"
echo "#               O(W) row) | oneTBB parallel_scan"
echo "#   cores     : 1 (HL_NUM_THREADS=1) and all (HL_NUM_THREADS=nproc)"
echo "#   consumer  : /(x+1) running mean, and >>2 (cheap, isolates the scan)"
echo "# All row-based W*H, parallel over rows, same data (fold dumps; the others"
echo "# validate against it). 3 variants x 2 core-counts x 2 consumers = 12 cells."
echo "############################################"
PXW=1048576
PXH=32
# Parallel axis = the H rows, parallel(y) -> tasks=H. Multicore point capped to H.
for cons in "div" "shr"; do
    [ "$cons" = shr ] && CENV="SHR=1" || CENV=""
    for nt in 1 "$(cap $PXH)"; do
        echo "--- consumer=$cons  threads=$nt (min(nproc,H=$PXH))  (W=$PXW H=$PXH) ---"
        # fold dumps the reference data for this (consumer,threads) cell first.
        run env $CENV HL_NUM_THREADS=$nt ./prefixsum_bench $PXW $PXH "$DATA_DIR/pfx.bin"
        # inductive UNFOLDED ablation (fold x -> W): same fusion, materializes the
        # full O(W) prefix row instead of the single accumulator -- isolates folding.
        run env UNFOLD=1 $CENV HL_NUM_THREADS=$nt ./prefixsum_bench $PXW $PXH "$DATA_DIR/pfx.bin"
        run env $CENV HL_NUM_THREADS=$nt ./prefixsum_bench_rdom $PXW $PXH "$DATA_DIR/pfx.bin"
        if [ -x ./prefixsum_bench_tbb ]; then
            run env $CENV HL_NUM_THREADS=$nt ./prefixsum_bench_tbb "$DATA_DIR/pfx.bin"
        fi
    done
done
run ./prefixsum_sweep 32 65536 262144 1048576

echo "############################################"
echo "# Prefix-scan parallelism regimes: one-stage (parallel over independent"
echo "# lanes only) vs oneTBB parallel_scan vs two-stage inductive scan"
echo "# (parallel ALONG time). Sweeps the lane count S at a fixed long length."
echo "# At small S the one-stage has no time-parallelism and loses to oneTBB;"
echo "# the two-stage recovers it and matches/beats oneTBB across S."
echo "############################################"
# Each Halide variant dumps the same {W,H,input,output} file (identical input
# formula), so oneTBB validates against whichever Halide variant ran last.
# one-stage parallelizes over the S lanes (tasks=S); two-stage over C=W/L chunks
# (tasks=W/4096). Each capped to its own task count.
for S in 1 8 32; do
    echo "--- W=1048576  S=$S ---"
    run env HL_NUM_THREADS=$(cap "$S") ./prefixsum_bench_onestage 1048576 "$S" "$DATA_DIR/prefixscan_data.bin"
    if [ -x ./prefixsum_bench_tbb ]; then
        run ./prefixsum_bench_tbb "$DATA_DIR/prefixscan_data.bin"
    fi
    run env HL_NUM_THREADS=$(cap $((1048576 / 4096))) ./prefixsum_bench_fold 1048576 "$S" 4096 "$DATA_DIR/prefixscan_data.bin"
done
# Same two-stage scan with a cheap >>2 consumer instead of the /(t+1) running
# mean, to isolate the scan/fold throughput from the (expensive) integer divide.
echo "--- scan/fold isolated (>>2 consumer), W=1048576 S=1 ---"
run env SHR=1 HL_NUM_THREADS=$(cap $((1048576 / 4096))) ./prefixsum_bench_fold 1048576 1 4096 "$DATA_DIR/prefixscan_shr.bin"

echo "############################################"
echo "# Chebyshev semi-iteration (inductive vs non-inductive solver)"
echo "# Dimension roles: n = PER-STEP work (O(n^2)/iter matvec), M = RECURRENCE"
echo "# length (folded axis, M -> 3). No batch axis. This solver's recurrence is its"
echo "# convergence count (~1e2), so M stays realistic and the footprint n*M*8 stays"
echo "# cache-resident -- the intended IN-CACHE datapoint (fold ~ tie) on the"
echo "# fp/LLC collapse plot, contrasting the out-of-cache sequence apps."
echo "############################################"
# chebyshev has NO parallel axis (serial solver, dense matvec vectorized only)
# -> 1 task -> 1 thread.
echo "### (A) per-step-work sweep  HL_NUM_THREADS=1 (serial: no parallel axis)  M=100 fixed,  n grows O(n^2) work ###"
for CN in 512 1024 2048 3072; do
    run env HL_NUM_THREADS=1 ./chebyshev_inductive "$CN" 100
done
echo "### (B) recurrence-length (fold-ratio) sweep  n=2048 fixed,  M grows fold ratio ###"
for CM in 50 100 200 400; do
    run ./chebyshev_inductive 2048 "$CM"
done

echo "############################################"
echo "# ODE integration: SPARSE Allen-Cahn reaction-diffusion (AB2), fold/unfold/non-inductive"
echo "# Dimension roles: D = PER-STEP work (spatial stencil width), T = RECURRENCE"
echo "# length (folded axis, n -> 3), B = BATCH (parallel axis). Footprint D*B*T*4."
echo "############################################"
# ode has NO parallel axis (the batch dim B is NOT scheduled parallel) -> serial
# -> 1 thread. NOTE: the (B) batch sweep therefore does not parallelize over B;
# if the paper wants a parallel batch story here, ode needs a .parallel(b) added.
echo "### (A) recurrence-length sweep  HL_NUM_THREADS=1 (serial: no parallel axis)  D=1024 B=1 fixed,  T crossing LLC ###"
for OT in 2048 8192 32768 131072; do
    run env HL_NUM_THREADS=1 ./ode_observer_sparse_fused_test 1024 1 "$OT"
done
echo "### (B) batch-size sweep  D=1024 T=8192 fixed,  B grows footprint (serial: no parallel-B) ###"
for BS in 1 4 16; do
    run env HL_NUM_THREADS=1 ./ode_observer_sparse_fused_test 1024 "$BS" 8192
done

echo "############################################"
echo "# Mamba / S6 selective scan (fused coefficients, CPU vs GPU)"
echo "############################################"
if [ -x ./mamba_ssm ]; then
    echo "# CPU: sweep sequence length T (D=16384 channels, N=16 state)"
    for T in 64 256 1024 4096; do
        run env T=$T ./mamba_ssm
    done
    echo "# CPU: sweep state dimension N at fixed T=512 (larger N = more work per token)"
    for N in 8 16 64; do
        run env N=$N T=512 ./mamba_ssm
    done
else
    echo "# (./mamba_ssm not present -- skipping Mamba CPU sweep)"
fi

# GPU is opt-in via USE_GPU=1 in mamba_ssm, and the binary itself does not probe
# for a CUDA device -- it will hard-fail at realize() if none is present. Only
# attempt the GPU path if a CUDA device is actually visible on this machine.
has_gpu=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
    has_gpu=1
elif [ -e /dev/nvidiactl ] || compgen -G "/dev/nvidia[0-9]*" >/dev/null 2>&1; then
    has_gpu=1
fi
if [ "$has_gpu" -eq 1 ]; then
    echo "# CUDA device detected -- running GPU path"
    # On WSL the CUDA driver (libcuda / cuInit) lives here, not a standard path;
    # harmless on native Linux where the driver is already on the loader path.
    [ -d /usr/lib/wsl/lib ] && export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # GPU: sweep sequence length T for the fused single-sequence scan.
    for T in 512 2048 8192; do
        run env USE_GPU=1 T=$T ./mamba_ssm
    done
    if [ -x ./mamba_gpu_batched ]; then
        # Batched selective scan (register-fold inductive scan): the parallel axis
        # is batch*D, so throughput should climb with batch as GPU occupancy fills.
        # With only 3 args the binary sweeps batch = 1,2,4,8,16,32 internally.
        echo "# -- occupancy sweep: realistic Mamba-2 config N=64 D=2048 T=8192, batch 1..16 --"
        run ./mamba_gpu_batched 64 2048 8192
        # A second geometry: smaller state N=16, wider channels, moderate length.
        echo "# -- occupancy sweep: N=16 D=4096 T=4096, batch 1..16 --"
        run ./mamba_gpu_batched 16 4096 4096
        # Inductive (register fold) vs non-inductive (materialize O(B*D*N*T) in
        # global memory): MAMBA_NONIND=1 adds the materialized row. Kept to a
        # single small batch so the materialized state fits in device memory.
        echo "# -- inductive vs non-inductive (materialize) at N=64 D=2048 T=2048 batch=1 --"
        run env MAMBA_NONIND=1 ./mamba_gpu_batched 64 2048 2048 1
    fi
else
    echo "# No CUDA device detected -- skipping GPU path (USE_GPU=1 would hard-fail)"
fi

echo "############################################"
echo "Summary: $pass passed, $fail failed"
echo "############################################"
exit $([ "$fail" -eq 0 ] && echo 0 || echo 1)
