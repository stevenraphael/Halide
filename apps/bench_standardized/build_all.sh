#!/usr/bin/env bash
# Build every standardized benchmark into the standalone_bins/ folder as a
# self-contained binary (bundled libHalide.so.22 / libtbb via $ORIGIN rpath, and
# statically-linked image-IO/OpenCV) so the suite runs on a fresh machine with
# NO Halide/OpenCV/TBB dev headers and (mostly) no runtime libs installed.
#
# Prereqs on THIS build box: distrib_install/ (Halide headers+lib), tools/GenGen.cpp,
# static libpng/jpeg/z + minimal static OpenCV (core/imgproc/calib3d/features2d/flann),
# boost headers (header-only, for the ODE baseline). None of these are needed on the
# RUN machine.
#
# Usage:  ./build_all.sh    (run from apps/bench_standardized)
set -euo pipefail
cd "$(dirname "$0")"
RB=$(cd ../.. && pwd)
SB=$RB/standalone_bins
INC=$RB/distrib_install/include
LIB=$RB/distrib_install/lib
TOOLS=$RB/distrib_install/share/tools
A=/usr/lib/x86_64-linux-gnu
# Cross-build knobs. Default to this machine (native/host). To build here for a
# DIFFERENT measurement machine, set both to that machine's CPU, e.g. an Ice Lake
# Xeon target from an Alder Lake build box:
#   ARCH=icelake-server \
#   HLTARGET=x86-64-linux-sse41-avx-avx2-fma-f16c-avx512-avx512_skylake-avx512_cannonlake \
#   ./build_all.sh
# ARCH  -> gcc -march for the driver/harness C++ (JIT kernels auto-detect the CPU
#          at runtime on the target, so they use its full ISA regardless).
# HLTARGET -> Halide target baked into the AOT stereobm libraries.
ARCH=${ARCH:-native}
HLTARGET=${HLTARGET:-host}
HFLAGS="-O3 -march=$ARCH -std=c++17 -fopenmp -I$INC"
LINKH="-L$SB -l:libHalide.so.22 -lpthread -ldl -Wl,-rpath,\$ORIGIN"

# libHalide bundle (source of truth = distrib_install; byte-identical copy)
cp -u $LIB/libHalide.so.22 $SB/ 2>/dev/null || true

echo "== pure-Halide JIT benches =="
for pair in \
    "viterbi_log.cpp:viterbi_log" \
    "ar_ll.cpp:kalman_ar" \
    "chebyshev_test.cpp:chebyshev_inductive" \
    "prefixsum_bench.cpp:prefixsum_bench" \
    "prefixsum_bench_rdom.cpp:prefixsum_bench_rdom" \
    "prefixsum_sweep.cpp:prefixsum_sweep" \
    "prefixsum_bench_onestage.cpp:prefixsum_bench_onestage" \
    "prefixsum_bench_fold.cpp:prefixsum_bench_fold"; do
    src=${pair%%:*}
    bin=${pair##*:}
    echo "  $src -> $bin"
    g++ $HFLAGS "$src" $LINKH -o "$SB/$bin"
done

echo "== ODE bench (Boost.odeint, header-only) =="
g++ $HFLAGS ode_observer_sparse_fused_test.cpp $LINKH -o "$SB/ode_observer_sparse_fused_test"

echo "== oneTBB comparison (bundle libtbb.so.12; no Halide) =="
cp -u $A/libtbb.so.12 $SB/
g++ -O3 -march=native -std=c++17 prefixsum_bench_tbb.cpp -ltbb -lpthread \
    -Wl,-rpath,'$ORIGIN' -o $SB/prefixsum_bench_tbb

echo "== GPU mamba selective scan (needs CUDA at run time; gated in run_tests.sh) =="
g++ $HFLAGS mamba_gpu_batched.cpp $LINKH -o "$SB/mamba_gpu_batched"
# Fused single-sequence selective scan (CPU always; GPU via USE_GPU=1). Dims D/N/T
# are env-overridable so run_tests.sh can sweep sequence length and state size.
g++ $HFLAGS ../mamba_ssm.cpp $LINKH -o "$SB/mamba_ssm"

echo "== GPU prefix scan: Halide inductive two-stage (JIT host-cuda) =="
g++ $HFLAGS prefixsum_gpu.cpp $LINKH -o "$SB/prefixsum_gpu"
echo "== GPU NON-ASSOCIATIVE scan: Halide inductive (no vendor equivalent) =="
g++ $HFLAGS prefixsum_gpu_nonassoc.cpp $LINKH -o "$SB/prefixsum_gpu_nonassoc"
# Third-party GPU baseline: NVIDIA Thrust/CUB segmented scan. Header-only; static
# cudart so the run machine needs only the CUDA driver. Skipped if no nvcc here.
if command -v nvcc >/dev/null 2>&1; then
    echo "== GPU prefix scan: Thrust/CUB baseline (nvcc, static cudart) =="
    nvcc -O3 -std=c++17 -arch=sm_89 --cudart static \
        prefixsum_gpu_thrust.cu -o "$SB/prefixsum_gpu_thrust"
else
    echo "  (nvcc not found; skipping prefixsum_gpu_thrust third-party baseline)"
fi

echo "== stereobm: AOT generators -> static libs -> drivers (window/tile sweep) =="
# winsize and tilesize are baked at AOT time (GeneratorParams) and, for OpenCV,
# at compile time (-DWINSIZE), so each (winsize,tilesize) config is its own pair
# of binaries. The baseline w9_t64 is also copied to the plain names that the
# small single-thread run in run_tests.sh uses; the larger configs feed the
# multicore full-resolution sweep. depth is held at 16 across configs so the
# OpenCV parity comparison stays apples-to-apples.
g++ -std=c++17 -O3 stereobm_generator.cpp $RB/tools/GenGen.cpp -I$INC -L$LIB -lHalide -lpthread -ldl -Wl,-rpath,$LIB -o /tmp/sb_ind.gen
g++ -std=c++17 -O3 stereobm_noninductive_generator.cpp $RB/tools/GenGen.cpp -I$INC -L$LIB -lHalide -lpthread -ldl -Wl,-rpath,$LIB -o /tmp/sb_ni.gen
CVINC=$(pkg-config --cflags opencv4 2>/dev/null || true)
CVLIBS="$A/libopencv_calib3d.a $A/libopencv_features2d.a $A/libopencv_flann.a $A/libopencv_imgproc.a $A/libopencv_core.a"
[ -f $A/libopencv_core.a ] && cp -u $A/libtbb.so.2 $SB/

# stereobm_jit: JIT Halide fold/unfold pipeline TIMED against OpenCV StereoBM on
# the real aloe images. Fast (no generator); OpenCV is linked directly (the
# generator is irrelevant to running OpenCV). Static OpenCV + png/jpeg/z, bundled
# libtbb.so.2 / libHalide.so.22 via $ORIGIN.
if [ -f $A/libopencv_core.a ]; then
    echo "  stereobm_jit.cpp -> stereobm_jit (with OpenCV timing)"
    g++ $HFLAGS -DSTEREOBM_BUILD_OPENCV=1 -I../support -I$TOOLS $CVINC stereobm_jit.cpp \
        -L$SB -l:libHalide.so.22 $CVLIBS $A/libpng.a $A/libjpeg.a $A/libz.a $SB/libtbb.so.2 \
        -lpthread -ldl -Wl,-rpath,'$ORIGIN' -o $SB/stereobm_jit
else
    echo "  (static OpenCV not found; building stereobm_jit WITHOUT OpenCV timing)"
    g++ $HFLAGS -I../support -I$TOOLS stereobm_jit.cpp $LINKH \
        $A/libpng.a $A/libjpeg.a $A/libz.a -o $SB/stereobm_jit
fi
# Sweep tilesize (strip width) ONLY; winsize is held fixed so the SAD window (and
# thus the OpenCV parity comparison) is identical across configs -- tilesize is a
# pure scheduling knob that doesn't change the result.
W=9
for TS in 32 64 128; do
    sfx="w${W}_t${TS}"
    echo "  -- stereobm config winsize=$W tilesize=$TS ($sfx) --"
    GP="winsize=$W depth=16 tilesize=$TS threshold=10 mindisp=0 uniqueness_ratio=0 filtercap=31"
    DEF="-DWINSIZE=$W -DDEPTH=16 -DPREFILTER_CAP=31 -DTEXTURE_THRESHOLD=10 -DUNIQUENESS_RATIO=0 -DMIN_DISP=0"
    # inductive carries the runtime; unfolded (fold=0) and non-inductive are no_runtime.
    LD_LIBRARY_PATH=$LIB /tmp/sb_ind.gen -g stereobm -e static_library,h -o . -f stereobm_inductive target=$HLTARGET $GP fold=true
    LD_LIBRARY_PATH=$LIB /tmp/sb_ind.gen -g stereobm -e static_library,h -o . -f stereobm_unfolded target=$HLTARGET-no_runtime $GP fold=false
    LD_LIBRARY_PATH=$LIB /tmp/sb_ni.gen -g stereobm -e static_library,h -o . -f stereobm_noninductive target=$HLTARGET-no_runtime $GP
    # non-OpenCV variant: static png/jpeg/z, no external deps
    g++ -std=c++17 -O3 -DSTEREOBM_BUILD_OPENCV=0 $DEF stereobm_process.cpp \
        stereobm_inductive.a stereobm_unfolded.a stereobm_noninductive.a -I. -I$INC -I$TOOLS \
        $A/libpng.a $A/libjpeg.a $A/libz.a -lpthread -ldl -o $SB/stereobm_process_$sfx
    # OpenCV comparison variant: static minimal OpenCV + bundled old libtbb.so.2
    if [ -f $A/libopencv_core.a ]; then
        g++ -std=c++17 -O3 -DSTEREOBM_BUILD_OPENCV=1 $DEF $CVINC stereobm_process.cpp \
            stereobm_inductive.a stereobm_unfolded.a stereobm_noninductive.a -I. -I$INC -I$TOOLS \
            $CVLIBS $A/libpng.a $A/libjpeg.a $A/libz.a $A/libtbb.so.2 -lpthread -ldl \
            -Wl,-rpath,'$ORIGIN' -o $SB/stereobm_process_opencv_$sfx
    else
        echo "  (static OpenCV not found; skipping stereobm_process_opencv_$sfx)"
    fi
    rm -f stereobm_inductive.a stereobm_inductive.h stereobm_unfolded.a stereobm_unfolded.h stereobm_noninductive.a stereobm_noninductive.h
done
# Baseline config keeps the plain names used by the small single-thread run.
cp -u $SB/stereobm_process_w9_t64 $SB/stereobm_process
[ -f $SB/stereobm_process_opencv_w9_t64 ] && cp -u $SB/stereobm_process_opencv_w9_t64 $SB/stereobm_process_opencv

# Full-resolution aloe pair for the multicore sweep (small pair stays for the
# quick single-thread run). Copied in so the folder is self-contained.
[ -f "$HOME/aloeL.png" ] && cp -u "$HOME/aloeL.png" $SB/aloeL_large.png
[ -f "$HOME/aloeR.png" ] && cp -u "$HOME/aloeR.png" $SB/aloeR_large.png

echo "== done; binaries in $SB =="
