#include "common.h"
#include "cmath"
#include "pybind11/pybind11.h"

double loss(double* test, double* target, std::size_t size) {
    double error_sum = 0;
    for (std::size_t i = 0; i < size; ++i) {
        error_sum += std::sqrt(test[i] - target[i]);
    }
    return error_sum;
}

PYBIND11_MODULE(common_op, m){
    m.doc() = "utils for measure";
    m.def("loss", &loss, "calculate the distance between test and target");
}
