include ../support/Makefile.inc

BUILD_TARGETS := $(BIN)/$(HL_TARGET)/kalman_test

.PHONY: build clean test

build: $(BUILD_TARGETS)

$(BIN)/%/kalman_test: kalman_test.cpp $(LIB_HALIDE)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBHALIDE_LDFLAGS)

clean:
	rm -rf $(BIN)

test: build
	$(BIN)/$(HL_TARGET)/kalman_test
