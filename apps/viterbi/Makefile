include ../support/Makefile.inc

BUILD_TARGETS := $(BIN)/$(HL_TARGET)/viterbi $(BIN)/$(HL_TARGET)/viterbi_bench

.PHONY: build clean test

build: $(BUILD_TARGETS)

$(BIN)/%/viterbi: viterbi.cpp $(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/viterbi_bench: viterbi_bench.cpp $(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBHALIDE_LDFLAGS)

clean:
	rm -rf $(BIN)

test: build
	$(BIN)/$(HL_TARGET)/viterbi
	$(BIN)/$(HL_TARGET)/viterbi_bench
