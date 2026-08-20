cmake_minimum_required(VERSION 3.28)
project(rnn_cascade)

enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Halide REQUIRED)

add_halide_generator(
    rnn_cascade.generator
    SOURCES rnn_cascade_generator.cpp
    LINK_LIBRARIES Halide::Tools
)

add_halide_library(
    rnn_cascade
    FROM rnn_cascade.generator
    ${RNN_CASCADE_FEATURES}
    PARAMS inductive=true
)
add_halide_library(
    rnn_cascade_noninductive
    FROM rnn_cascade.generator
    GENERATOR rnn_cascade
    ${RNN_CASCADE_FEATURES}
    PARAMS inductive=false
)

add_executable(rnn_cascade_test test.cpp)
target_link_libraries(rnn_cascade_test PRIVATE Halide::Tools rnn_cascade rnn_cascade_noninductive)

add_test(NAME rnn_cascade_test COMMAND rnn_cascade_test)
set_tests_properties(
    rnn_cascade_test
    PROPERTIES
    LABELS rnn_cascade
    PASS_REGULAR_EXPRESSION "Success!"
    SKIP_REGULAR_EXPRESSION "\\[SKIP\\]"
)
