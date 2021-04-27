set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
        CACHE STRING "Vcpkg toolchain file")

project(matmul-conv)
cmake_minimum_required(VERSION 3.10)

add_subdirectory(pybind11)

set(build-cpu ON)
set(build-gpu ON)
set(build-mlu OFF)

if (${build-cpu})
    add_subdirectory(CPU)
endif ()

if (${build-gpu})
    add_subdirectory(GPU)
endif ()

if (${build-mlu})
    add_subdirectory(MLU)
endif ()
