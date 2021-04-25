project(matmul-conv)
cmake_minimum_required(VERSION 3.10)

add_subdirectory(pybind11)
add_subdirectory(CPU)
add_subdirectory(GPU)
add_subdirectory(MLU)