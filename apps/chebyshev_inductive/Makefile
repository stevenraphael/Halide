include ../support/Makefile.inc

# Self-contained JIT app: the Chebyshev pipeline is built and JIT-compiled inside
# test.cpp, so there is no generator/AOT step to run -- just compile and link
# against libHalide like any JIT program.

.PHONY: build clean test

build: $(BIN)/$(HL_TARGET)/test

$(BIN)/%/test: test.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -Wall -O3 $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

clean:
	rm -rf $(BIN)

test: $(BIN)/$(HL_TARGET)/test
	$<
