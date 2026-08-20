# Chebyshev + Kalman re-run bundle (prebuilt)

Self-contained re-run of the two benchmarks whose sources changed
(channel-RVar unroll in Kalman; the extra no-modulo Chebyshev variant). Both now
report byte-exact **measured** allocations via the custom allocator.

Same shipping model as `standalone_bins/`: the binaries are **prebuilt** on the
build box with `libHalide.so.22` bundled via `$ORIGIN` rpath, so the measurement
machine needs **NO Halide headers, LLVM, or build tools** -- just copy this whole
folder over and run.

## Contents
```
run.sh                     run both sweeps from the prebuilt binaries
kalman_ar                  prebuilt (rc/r.x unrolled + measured allocs)
chebyshev_inductive        prebuilt (4 variants incl. no-modulo materialize)
libHalide.so.22            bundled runtime (found via $ORIGIN rpath)
src/                       sources, for reference only (not needed to run)
support/                   headers, for reference only
```

## Run it (on the measurement machine)
```bash
./run.sh                       # default protocol HB_TRIALS=30 HB_WARMUP=3
```
Output: `kalman.txt`, `chebyshev.txt`, `cheb_kalman.txt` in this folder. Each row
shows median time, throughput, and `state_bytes=<N>` (measured peak heap). Feed
these back to the plotting scripts (they replace only the Kalman/Chebyshev rows).

## If it dies with SIGILL (exit 132)
The binaries were built with `-march=native` for the build box's CPU. If this
machine is a different/older CPU, rebuild on the build box for the target ISA and
re-copy. On the build box (with `distrib_install/`):
```bash
D=../../distrib_install
g++ -O3 -march=<target-cpu> -std=c++17 -fopenmp -I$D/include src/ar_ll.cpp \
    -L. -l:libHalide.so.22 -lpthread -ldl -Wl,-rpath,'$ORIGIN' -o kalman_ar
g++ -O3 -march=<target-cpu> -std=c++17 -fopenmp -I$D/include src/chebyshev_test.cpp \
    -L. -l:libHalide.so.22 -lpthread -ldl -Wl,-rpath,'$ORIGIN' -o chebyshev_inductive
```
(The JIT kernels auto-detect the CPU at runtime; only the harness C++ bakes in
`-march`.)
