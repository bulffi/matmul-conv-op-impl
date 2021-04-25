add_subdirectory(eigen)

pybind11_add_module(cpu_op cpu_op.cpp)

add_executable(eigen_matmul eigen_op.cpp)
target_link_libraries(eigen_matmul eigen)
target_compile_features(eigen_matmul PRIVATE cxx_std_17)