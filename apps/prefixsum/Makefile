include ../support/Makefile.inc

# prefixsum_bench_tbb.cpp compares against oneTBB's parallel_scan.
# Auto-disable if oneTBB is not available via pkg-config.
HAVE_TBB := $(shell pkg-config --exists tbb 2>/dev/null && echo yes)
PREFIXSUM_BUILD_TBB ?= $(if $(HAVE_TBB),1,0)
ifeq ($(PREFIXSUM_BUILD_TBB),1)
  TBB_CXXFLAGS := $(shell pkg-config --cflags tbb 2>/dev/null)
  TBB_LIBS := $(shell pkg-config --libs tbb 2>/dev/null)
endif

BUILD_TARGETS := $(BIN)/$(HL_TARGET)/prefixsum_bench \
                  $(BIN)/$(HL_TARGET)/prefixsum_bench_rdom \
                  $(BIN)/$(HL_TARGET)/prefixsum_sweep
ifeq ($(PREFIXSUM_BUILD_TBB),1)
  BUILD_TARGETS += $(BIN)/$(HL_TARGET)/prefixsum_bench_tbb
endif

.PHONY: build clean test

build: $(BUILD_TARGETS)

$(BIN)/%/prefixsum_bench: prefixsum_bench.cpp $(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/prefixsum_bench_rdom: prefixsum_bench_rdom.cpp $(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/prefixsum_sweep: prefixsum_sweep.cpp $(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBHALIDE_LDFLAGS)

# prefixsum_bench_tbb.cpp is a standalone (non-Halide) benchmark, compiled
# without the Halide include/link flags.
$(BIN)/%/prefixsum_bench_tbb: prefixsum_bench_tbb.cpp
	@mkdir -p $(@D)
	$(CXX) $(OPTIMIZE) -std=c++17 $(TBB_CXXFLAGS) $< -o $@ $(TBB_LIBS)

clean:
	rm -rf $(BIN)

# prefixsum_bench must run first: it dumps the data file the others compare
# against.
test: build
	$(BIN)/$(HL_TARGET)/prefixsum_bench 65536 32 /tmp/prefixsum_bench_data.bin
	$(BIN)/$(HL_TARGET)/prefixsum_bench_rdom 65536 32 /tmp/prefixsum_bench_data.bin
ifeq ($(PREFIXSUM_BUILD_TBB),1)
	$(BIN)/$(HL_TARGET)/prefixsum_bench_tbb /tmp/prefixsum_bench_data.bin
endif
	$(PYTHON) bench_numpy.py /tmp/prefixsum_bench_data.bin
