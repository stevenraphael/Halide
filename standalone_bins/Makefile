# Build + run the standardized inductive-fold benchmark suite on THIS machine.
#
# Prereq: a built/installed Halide distrib (headers + libHalide.so.22). Point at
# it with HALIDE_DISTRIB (defaults to ../distrib_install, i.e. a distrib sitting
# next to this checkout). Nothing else is prebuilt -- `make` compiles every
# benchmark from ../apps/bench_standardized against that distrib into THIS folder,
# then `make run` executes run_tests.sh (which finds the binaries + aloe images
# here, and libHalide via the rpath baked in below).
#
# Usage:
#   make HALIDE_DISTRIB=/path/to/distrib_install     # build the core CPU suite
#   make run                                         # run the full test script
#   make gpu                                         # (optional) CUDA benches
#   make clean
#
# OpenCV (for the stereobm timing baseline) and oneTBB (for the prefix-sum
# baseline) are used automatically IF their dev packages are found via pkg-config;
# otherwise those two binaries are skipped and the suite still runs (run_tests.sh
# guards missing binaries).

HALIDE_DISTRIB ?= $(abspath ../distrib_install)
INC   := $(HALIDE_DISTRIB)/include
LIB   := $(HALIDE_DISTRIB)/lib
TOOLS := $(HALIDE_DISTRIB)/share/tools
SRC   := ../apps/bench_standardized
SUP   := ../apps/support

CXX      ?= g++
ARCH     ?= native
CXXFLAGS := -O3 -march=$(ARCH) -std=c++17 -fopenmp -I$(INC) -I$(SUP) -I$(TOOLS)
# rpath -> the distrib lib dir, so the binaries find libHalide.so.22 at runtime
# without copying it here (it is intentionally NOT committed to this branch).
LDFLAGS  := -L$(LIB) -l:libHalide.so.22 -lpthread -ldl -Wl,-rpath,$(LIB)

# Image IO (stereobm loads the aloe PNGs via Halide's image_io).
IMGIO := -ljpeg -lpng -lz

# Optional deps, auto-detected.
HAVE_OPENCV := $(shell pkg-config --exists opencv4 2>/dev/null && echo 1)
HAVE_TBB    := $(shell (echo '\#include <tbb/tbb.h>' | $(CXX) -E -x c++ - >/dev/null 2>&1) && echo 1)

# Core CPU benchmarks that run_tests.sh exercises (source -> binary name).
CORE := viterbi_log kalman_ar chebyshev_inductive \
        prefixsum_bench prefixsum_bench_rdom prefixsum_bench_onestage \
        prefixsum_bench_fold prefixsum_sweep \
        ode_observer_sparse_fused_test mamba_ssm

OPT :=
ifeq ($(HAVE_TBB),1)
OPT += prefixsum_bench_tbb
endif
ifneq ($(strip $(wildcard $(SRC)/stereobm_jit.cpp)),)
OPT += stereobm_jit
endif

.PHONY: all core opt gpu run clean check-distrib
all: check-distrib core opt
	@echo "== built: $(CORE) $(OPT) =="
	@[ "$(HAVE_OPENCV)" = 1 ] || echo "   (OpenCV not found -> stereobm_jit built without the OpenCV timing baseline)"
	@[ "$(HAVE_TBB)" = 1 ]    || echo "   (oneTBB not found -> prefixsum_bench_tbb skipped)"
core: $(CORE)
opt: $(OPT)

check-distrib:
	@test -f "$(INC)/Halide.h" || { \
	  echo "ERROR: Halide.h not found under HALIDE_DISTRIB=$(HALIDE_DISTRIB)"; \
	  echo "       Point HALIDE_DISTRIB at a built Halide distrib, e.g."; \
	  echo "       make HALIDE_DISTRIB=/path/to/Halide/distrib_install"; \
	  exit 1; }
	@test -f "$(LIB)/libHalide.so.22" || { \
	  echo "ERROR: libHalide.so.22 not found under $(LIB)"; exit 1; }

# --- source name != binary name, so give those explicit rules ---
kalman_ar: $(SRC)/ar_ll.cpp $(SUP)/bench_harness.h
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@
chebyshev_inductive: $(SRC)/chebyshev_test.cpp $(SUP)/bench_harness.h
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@
mamba_ssm: ../apps/mamba_ssm.cpp $(SUP)/bench_harness.h
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

# --- source name == binary name (the common case) ---
%: $(SRC)/%.cpp $(SUP)/bench_harness.h
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

# oneTBB prefix-sum baseline (no Halide).
prefixsum_bench_tbb: $(SRC)/prefixsum_bench_tbb.cpp
	$(CXX) -O3 -march=$(ARCH) -std=c++17 $< -ltbb -lpthread -o $@

# stereobm JIT: image IO always; OpenCV timing baseline if available.
ifeq ($(HAVE_OPENCV),1)
stereobm_jit: $(SRC)/stereobm_jit.cpp $(SUP)/bench_harness.h
	$(CXX) $(CXXFLAGS) -DSTEREOBM_BUILD_OPENCV=1 $(shell pkg-config --cflags opencv4) \
	    $< $(LDFLAGS) $(shell pkg-config --libs opencv4) $(IMGIO) -o $@
else
stereobm_jit: $(SRC)/stereobm_jit.cpp $(SUP)/bench_harness.h
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) $(IMGIO) -o $@
endif

# --- GPU benches (prefix-sum + mamba) ---
# The Halide ones JIT PTX at runtime, so they build with just g++ + libHalide and
# need only the CUDA driver present to RUN. The Thrust/CUB prefix-sum baseline is
# pure CUDA and needs nvcc at build time. SM_ARCH must match the run GPU's compute
# capability (A10G = sm_86); the Halide sources also bake CUDACapability86.
SM_ARCH     ?= sm_86
HAVE_NVCC   := $(shell command -v nvcc >/dev/null 2>&1 && echo 1)
# Official mamba v1 selective-scan CUDA source (git clone state-spaces/mamba);
# override if you cloned elsewhere. Only needed for the mamba vendor baseline.
MAMBA_SRC   ?= $(HOME)/mamba_src/csrc/selective_scan
GPU_HALIDE  := prefixsum_gpu prefixsum_gpu_nonassoc mamba_gpu_batched mamba_gpu_2stage
GPU_VENDOR  :=
ifeq ($(HAVE_NVCC),1)
GPU_VENDOR  += prefixsum_gpu_thrust
ifneq ($(wildcard $(MAMBA_SRC)/selective_scan_fwd_kernel.cuh),)
GPU_VENDOR  += mamba_v1_bench
endif
endif

gpu: check-distrib $(GPU_HALIDE) $(GPU_VENDOR)
	@echo "== built GPU benches: $(GPU_HALIDE) $(GPU_VENDOR) =="
	@[ "$(HAVE_NVCC)" = 1 ] || echo "   (nvcc not found -> Thrust/CUB prefix-sum baseline skipped)"
	@echo "   prefixsum:  ./prefixsum_gpu 1048576 8 4096 /tmp/pfx.bin && ./prefixsum_gpu_thrust /tmp/pfx.bin"
	@echo "   mamba    :  USE_GPU=1 D=2048 N=16 T=8192 ./mamba_ssm ; ./mamba_gpu_batched 64 2048 8192"
	@echo "   mamba vendor: ./mamba_v1_bench 2048 16 8192 8   (CHECK=1 to validate)"
	@[ -n "$(wildcard $(MAMBA_SRC)/selective_scan_fwd_kernel.cuh)" ] || echo "   (mamba_v1_bench skipped: clone state-spaces/mamba to $(MAMBA_SRC) or set MAMBA_SRC)"

$(GPU_HALIDE): %: $(SRC)/%.cpp $(SUP)/bench_harness.h
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

# Thrust/CUB vendor baseline (needs CUDA toolkit; static cudart so only the driver
# is needed at run time).
prefixsum_gpu_thrust: $(SRC)/prefixsum_gpu_thrust.cu
	nvcc -O3 -std=c++17 -arch=$(SM_ARCH) --cudart static $< -o $@

# Official mamba v1 selective-scan CUDA kernel, driven torch-free (c10 shimmed).
# Needs the mamba source cloned at MAMBA_SRC.
mamba_v1_bench: $(SRC)/mamba_v1_bench.cu
	nvcc -O3 -std=c++17 -arch=$(SM_ARCH) --expt-relaxed-constexpr \
	  -I $(MAMBA_SRC) -I $(SRC)/mamba_v1_shims -I $(SUP) $< -o $@

run: all
	./run_tests.sh

clean:
	rm -f $(CORE) $(OPT) $(GPU_HALIDE) prefixsum_gpu_thrust mamba_v1_bench
