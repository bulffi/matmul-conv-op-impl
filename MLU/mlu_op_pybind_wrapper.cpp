#include "mlu_op.h"
#include <iostream>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

PYBIND11_MODULE(mlu_op, m){
    m.doc() = "matmul & conv with mlu";
    m.def("mlu_matmul_base", [](const pybind11::array_t<double>& M, const pybind11::array_t<double>& N) -> pybind11::array_t<double> {
        auto* input_1 = new float[100];
        auto* input_2 = new float[100];
        auto* result = new float[100];
        for(int i = 0;  i < 100; i++) {
            input_1[i] = i - 1;
            input_2[i] = i + 1;
        }
        MLUPowerDifferenceOp(input_1, input_2, 2, result, 100);
        pybind11::capsule free_when_done(result, [](void *f) {
            auto *foo = reinterpret_cast<float *>(f);
            delete[] foo;
        });
        delete[] input_1;
        delete[] input_2;
        return pybind11::array_t<float> {
            100,
            result,
            free_when_done
        };
    }, "multiply 2 matrix with mlu");
}