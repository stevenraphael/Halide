include ../support/Makefile.inc

all: $(BIN)/$(HL_TARGET)/merge_test

$(GENERATOR_BIN)/merge.generator: merge_generator.cpp $(GENERATOR_DEPS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -g $(filter %.cpp,$^) -o $@ $(LIBHALIDE_LDFLAGS)

$(BIN)/%/merge.a: $(GENERATOR_BIN)/merge.generator
	@mkdir -p $(@D)
	$< -g merge -f merge -e $(GENERATOR_OUTPUTS) -o $(BIN)/$* target=$*

$(BIN)/%/merge_test: merge_test.cpp $(BIN)/%/merge.a
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(BIN)/$* -Wall $^ -o $@ $(LDFLAGS)

test: $(BIN)/$(HL_TARGET)/merge_test
	$(BIN)/$(HL_TARGET)/merge_test

clean:
	rm -rf $(BIN)
