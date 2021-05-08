#include "mlu_op.h"
#include <iostream>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include <chrono>


pybind11::array_t<float> mlu_matmul_base_wrapper(const pybind11::array_t<float>& M, const pybind11::array_t<float>& N) {
    auto m = M.unchecked<2>();
    auto n = N.unchecked<2>();
    std::size_t m_x = m.shape(0);
    std::size_t m_y = m.shape(1);
    std::size_t n_x = n.shape(0);
    std::size_t n_y = n.shape(1);
    std::size_t m_size = m_x * m_y;
    std::size_t n_size = n_x * n_y;
    std::size_t output_size = m_x * n_y;
    auto *result = new float[output_size];
    
    /// ==============================================
    int code = mlu_matmul_base(m.data(0, 0), n.data(0, 0), result, m_x, m_y, n_y);
    /// ===============================================

    pybind11::capsule free_when_done(result, [](void *f) {
        auto *foo = reinterpret_cast<float *>(f);
        delete[] foo;
    });

    return pybind11::array_t<float> {
        {m_x, n_y},
        result,
        free_when_done
    };
}

PYBIND11_MODULE(mlu_op, m){
    m.doc() = "matmul & conv with mlu";
    m.def("mlu_matmul_base", &mlu_matmul_base_wrapper, "multiply 2 matrix with mlu");
}