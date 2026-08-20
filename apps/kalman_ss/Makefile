include ../support/Makefile.inc

KALMAN_SS_CXXFLAGS := $(CXXFLAGS) -fopenmp

.PHONY: build clean test

build: $(BIN)/$(HL_TARGET)/kalman_ss

$(BIN)/%/kalman_ss: kalman_ss.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(KALMAN_SS_CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

clean:
	rm -rf $(BIN)

test: build
	$(BIN)/$(HL_TARGET)/kalman_ss
