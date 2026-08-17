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

# Optional CUDA benches (need nvcc / a CUDA device at run time).
gpu: check-distrib mamba_gpu_batched prefixsum_gpu prefixsum_gpu_nonassoc
	@echo "== built GPU benches (run with a CUDA device present) =="
mamba_gpu_batched prefixsum_gpu prefixsum_gpu_nonassoc: %: $(SRC)/%.cpp $(SUP)/bench_harness.h
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

run: all
	./run_tests.sh

clean:
	rm -f $(CORE) $(OPT) mamba_gpu_batched prefixsum_gpu prefixsum_gpu_nonassoc
