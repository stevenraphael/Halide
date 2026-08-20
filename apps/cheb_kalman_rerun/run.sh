#!/usr/bin/env bash
# Re-run ONLY the Kalman AR(2) and Chebyshev benchmarks from PREBUILT binaries.
# Self-contained: kalman_ar / chebyshev_inductive find the bundled libHalide.so.22
# via their $ORIGIN rpath, so the run machine needs NO Halide headers, LLVM, or
# build tools -- just copy this folder over and run this script.
#
# (The binaries were built with -march=native on the build box. If this machine
# has a different/older CPU they die with SIGILL -- see the preflight below.)
#
# Usage:
#   ./run.sh                       # default protocol HB_TRIALS=30 HB_WARMUP=3
#   HB_TRIALS=30 HB_WARMUP=3 ./run.sh
#
# Output: kalman.txt, chebyshev.txt, cheb_kalman.txt (combined) in this folder.
set -u
cd "$(dirname "$0")"
chmod +x kalman_ar chebyshev_inductive 2>/dev/null || true

for b in kalman_ar chebyshev_inductive libHalide.so.22; do
    [ -e "./$b" ] || { echo "!!! missing ./$b -- copy the whole folder over"; exit 1; }
done

# Preflight: a binary built for a different CPU dies with SIGILL (exit 132).
./kalman_ar 4 8 >/dev/null 2>&1
if [ "$?" -eq 132 ]; then
    echo "!!! Illegal instruction (SIGILL): these binaries were built with"
    echo "!!! -march=native for a DIFFERENT CPU than this one. Rebuild on the"
    echo "!!! build box with ARCH set to THIS machine's CPU and re-copy."
    exit 1
fi

export HB_TRIALS="${HB_TRIALS:-30}" HB_WARMUP="${HB_WARMUP:-3}"
NCORES="$(nproc 2>/dev/null || echo 1)"
cap() { local n=$1; [ "$n" -lt "$NCORES" ] && echo "$n" || echo "$NCORES"; }
run() { echo "=== $* ==="; "$@"; echo; }

kalman() {
  echo "############################################"
  echo "# Kalman latent AR(2) log-likelihood (channel RVar UNROLLED, both paths)"
  echo "############################################"
  for nt in 1 "$(cap $((256 / 8)))"; do
    echo "### (A) recurrence-length  HL_NUM_THREADS=$nt  B=256, T crossing LLC ###"
    for T in 1024 4096 16384 65536; do run env HL_NUM_THREADS=$nt ./kalman_ar 256 "$T"; done
  done
  echo "### (B) batch  T=16384 fixed, B crossing LLC ###"
  for B in 16 64 256 1024; do run env HL_NUM_THREADS=$(cap $((B / 8))) ./kalman_ar "$B" 16384; done
  echo "### (C) arithmetic-intensity flip  B=256 T=16384 ###"
  run env HL_NUM_THREADS=$(cap $((256 / 8))) ./kalman_ar 256 16384
  run env HB_CHEAP=1 HL_NUM_THREADS=$(cap $((256 / 8))) ./kalman_ar 256 16384
}

chebyshev() {
  echo "############################################"
  echo "# Chebyshev semi-iteration (4 variants incl. no-modulo full materialize)"
  echo "############################################"
  echo "### (A) per-step-work  HL_NUM_THREADS=1  M=100, n grows ###"
  for CN in 512 1024 2048 3072; do run env HL_NUM_THREADS=1 ./chebyshev_inductive "$CN" 100; done
  echo "### (B) recurrence-length (fold-ratio)  n=2048, M grows ###"
  for CM in 50 100 200 400; do run ./chebyshev_inductive 2048 "$CM"; done
}

kalman    2>&1 | tee kalman.txt
chebyshev 2>&1 | tee chebyshev.txt
cat kalman.txt chebyshev.txt > cheb_kalman.txt
echo "# wrote kalman.txt, chebyshev.txt, cheb_kalman.txt"
