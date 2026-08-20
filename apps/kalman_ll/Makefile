include ../support/Makefile.inc

KALMAN_LL_CXXFLAGS := $(CXXFLAGS) -fopenmp

.PHONY: build clean test

build: $(BIN)/$(HL_TARGET)/kalman_ll $(BIN)/$(HL_TARGET)/kalman_ll_llt $(BIN)/$(HL_TARGET)/kalman_traj

$(BIN)/%/kalman_ll: kalman_ll.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(KALMAN_LL_CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/kalman_ll_llt: kalman_ll_llt.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(KALMAN_LL_CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/kalman_traj: kalman_traj.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(KALMAN_LL_CXXFLAGS) $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

clean:
	rm -rf $(BIN)

test: build
	$(BIN)/$(HL_TARGET)/kalman_ll
	$(BIN)/$(HL_TARGET)/kalman_ll_llt
	$(BIN)/$(HL_TARGET)/kalman_traj
