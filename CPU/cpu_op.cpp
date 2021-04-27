//
// Created by 张梵 on 2021/4/25.
//

#include <iostream>
#include "cpu_op.h"

pybind11::array_t<double> cpu_matmul_base(const pybind11::array_t<double>& M, const pybind11::array_t<double>& N) {
    auto m = M.unchecked<2>();
    auto n = N.unchecked<2>();
    std::size_t m_x = m.shape(0);
    std::size_t m_y = m.shape(1);
    std::size_t n_x = n.shape(0);
    std::size_t n_y = n.shape(1);
    std::size_t output_size = m_x * n_y;
    auto *result = new double[output_size];

    /// =========================================

    for (int i = 0; i < m_x; ++i) {
        for (int j = 0; j < n_y; ++j) {
            double tempt_sum = 0;
            for (int k = 0; k < m_y; ++k) {
                tempt_sum += m(i, k) * n(k, j);
            }
            result[i * n.shape(1) + j] = tempt_sum;
        }
    }

    /// =========================================

    pybind11::capsule free_when_done(result, [](void *f) {
        auto *foo = reinterpret_cast<double *>(f);
        delete[] foo;
    });
    return pybind11::array_t<double> {
            {m_x, n_y},
            result,
            free_when_done
    };
}

PYBIND11_MODULE(cpu_op, m){
    m.doc() = "matmul & conv with cpu";
    m.def("cpu_matmul_base", &cpu_matmul_base, "multiply 2 matrix");
}
