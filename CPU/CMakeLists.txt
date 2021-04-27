# find_package(Eigen3 CONFIG REQUIRED)
pybind11_add_module(cpu_op cpu_op.cpp)

# add_executable(eigen_matmul eigen_op.cpp)
# target_link_libraries(eigen_matmul PRIVATE Eigen3::Eigen)
# target_compile_features(eigen_matmul PRIVATE cxx_std_17)