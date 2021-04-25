//
// Created by 张梵 on 2021/4/25.
//

#include <iostream>
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
    constexpr size_t size = 100*1000*1000;
    auto *foo = new double[size];
    for (size_t i = 0; i < size; i++) {
        foo[i] = (double) i;
    }

    // Create a Python object that will free the allocated
    // memory when destroyed:
    pybind11::capsule free_when_done(foo, [](void *f) {
        auto *foo = reinterpret_cast<double *>(f);
        std::cerr << "Element [0] = " << foo[0] << "\n";
        std::cerr << "freeing memory @ " << f << "\n";
        delete[] foo;
    });

    return pybind11::array_t<double>(
            {100, 1000, 1000}, // shape
            {1000*1000*8, 1000*8, 8}, // C-style contiguous strides for double
            foo, // the data pointer
            free_when_done); // numpy array references this parent
}

PYBIND11_MODULE(cpu_op, m){
    m.doc() = "matmul & conv with cpu";
    m.def("cpu_matmul", &cpu_matmul, "multiply 2 matrix");
    m.def("add", &add, "A strange addition");
}
