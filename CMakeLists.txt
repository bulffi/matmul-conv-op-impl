project(matmul-conv)
cmake_minimum_required(VERSION 3.10)

add_subdirectory(pybind11)

set(build-cpu ON)
set(build-gpu OFF)
set(build-mlu OFF)

add_subdirectory(common)

if (${build-cpu})
    add_subdirectory(CPU)
endif ()

if (${build-gpu})
    add_subdirectory(GPU)
endif ()

if (${build-mlu})
    add_subdirectory(MLU)
endif ()
