cmake_minimum_required(VERSION 3.28)
project(chebyshev_inductive)

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Self-contained JIT app: the pipeline is built and JIT-compiled inside test.cpp,
# so no generator / add_halide_library step is needed -- just link libHalide.
add_executable(chebyshev_inductive_test test.cpp)
target_link_libraries(chebyshev_inductive_test PRIVATE Halide::Halide)

# Test that the app actually works!
add_test(NAME chebyshev_inductive_test COMMAND chebyshev_inductive_test)
set_tests_properties(
    chebyshev_inductive_test
    PROPERTIES
    LABELS chebyshev_inductive
    PASS_REGULAR_EXPRESSION "Success!"
    SKIP_REGULAR_EXPRESSION "\\[SKIP\\]"
)
