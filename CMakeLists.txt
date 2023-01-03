cmake_minimum_required(VERSION 3.24)
project(HyTorch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(Halide REQUIRED)

add_executable(OpGenerator generator.cpp)
target_link_libraries(OpGenerator PRIVATE Halide::Generator)

add_halide_library(AddOp FROM OpGenerator
                   GENERATOR add
                   PARAMS input_a.type=float32 input_b.type=float32
                   PYTORCH_WRAPPER _ignored_result)
