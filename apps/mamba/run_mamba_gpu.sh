#!/usr/bin/env bash
# Unified Mamba GPU head-to-head: Halide inductive/non-inductive variants vs the
# production fused CUDA kernel (mamba_ssm), at matched (N,D,T).
#
# Row source:
#   mamba_gpu_bench (this dir)  -> Halide (A) seq non-ind / (B) seq ind /
#                                  (C) two-stage non-ind / (D) two-stage ind
#   mamba_ssm_bench.py          -> SOTA fused CUDA selective_scan_fn
#
# The Halide driver reports median + IQR over 30 trials with an in-band device
# sync and a correctness gate; the non-inductive rows are expected to fail to
# allocate once D*N*T no longer fits (the "impossible without folding" point).
#
# Usage: HALIDE=/home/sraphael/Halide ./run_mamba_gpu.sh
set -u
HALIDE="${HALIDE:-/home/sraphael/Halide}"
export LD_LIBRARY_PATH="$HALIDE/build/src:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
BIN=/tmp/mamba_gpu_bench
SRC="$HALIDE/apps/mamba/mamba_gpu_bench.cpp"

if [ ! -x "$BIN" ] || [ "$SRC" -nt "$BIN" ]; then
    echo "building $BIN ..."
    g++ "$SRC" -O3 -march=native -I"$HALIDE/include" -L"$HALIDE/build/src" \
        -lHalide -lpthread -ldl -o "$BIN" -std=c++17 || exit 1
fi

# (N D T) points. Small/medium fit the non-inductive trajectory; large ones do
# not, which is the point.
SIZES=("16 512 16384" "16 512 65536" "16 1024 16384")

for s in "${SIZES[@]}"; do
    echo "############################################################"
    echo "# N D T = $s"
    echo "############################################################"
    # Halide variants (DEBUG chatter goes to stderr; keep the table only).
    "$BIN" $s 2>/dev/null
    # SOTA row: production Mamba-2 SSD kernel (mamba_chunk_scan_combined, Triton
    # JIT). We use this rather than the v1 selective_scan CUDA extension because
    # the prebuilt .so is ABI-incompatible with the installed torch (segfaults).
    if python3 -c "import mamba_ssm" 2>/dev/null; then
        echo "  ---- SOTA (third-party optimized) ----"
        python3 "$HALIDE/apps/mamba/mamba_ssd_bench.py" $s 2>/dev/null \
            || echo "  mamba_ssm SSD: [unavailable — see mamba_ssm/torch ABI]"
    fi
    echo
done
