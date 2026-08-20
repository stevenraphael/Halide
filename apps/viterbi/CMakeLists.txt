cmake_minimum_required(VERSION 3.28)
project(viterbi)

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_halide_generator(viterbi.generator SOURCES viterbi_generator.cpp)

# Filters: the default (use_inductive=true) build, and a build with the
# inductive-function machinery disabled in favor of a plain RDom scan.
add_halide_library(viterbi FROM viterbi.generator)
add_halide_library(
    viterbi_no_inductive
    FROM viterbi.generator
    GENERATOR viterbi
    PARAMS use_inductive=false
)

# Main executable
add_executable(viterbi_test test.cpp)
target_link_libraries(viterbi_test PRIVATE Halide::Tools viterbi viterbi_no_inductive)

# Test that the app actually works!
add_test(NAME viterbi_app COMMAND viterbi_test)
set_tests_properties(
    viterbi_app
    PROPERTIES
    LABELS viterbi
    PASS_REGULAR_EXPRESSION "Success!"
    SKIP_REGULAR_EXPRESSION "\\[SKIP\\]"
)
