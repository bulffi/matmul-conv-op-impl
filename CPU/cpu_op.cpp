//
// Created by 张梵 on 2021/4/25.
//

#include "cpu_op.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

double* cpu_matmul(double* input, double* weight, std::size_t H, std::size_t K, std::size_t W){
    return nullptr;
}
double* cpu_conv2d(double* input, double* weight, std::size_t batch, std::size_t H, std::size_t W,
                   std::size_t C, std::size_t w_H, std::size_t w_W, std::size_t w_batch) {
    return nullptr;
}

int add(int a, int b) {
    return a + b + 1;
}

pybind11::array_t<double> return_array() {

}

PYBIND11_MODULE(cpu_op, m){
    m.doc() = "matmul & conv with cpu";
    m.def("cpu_matmul", &cpu_matmul, "multiply 2 matrix");
    m.def("add", &add, "A strange addition");
}
