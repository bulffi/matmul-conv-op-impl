#include <cstdio>
#include "gpu_op.h"

__global__ void from_cuda(const double* M, const double* N, double* out, int size, int n_y) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += M[threadIdx.x * size + i] * N[i * n_y + threadIdx.y];
    }
    out[threadIdx.x * n_y + threadIdx.y] = sum;
}

pybind11::array_t<double> gpu_matmul_base(pybind11::array_t<double> M, pybind11::array_t<double> N) {
    auto m = M.unchecked<2>();
    auto n = N.unchecked<2>();
    std::size_t m_x = m.shape(0);
    std::size_t m_y = m.shape(1);
    std::size_t n_x = n.shape(0);
    std::size_t n_y = n.shape(1);
    std::size_t m_size = m_x * m_y;
    std::size_t n_size = n_x * n_y;
    std::size_t output_size = m_x * n_y;
    auto *result = new double[output_size];

    /// =========================================
    double* d_M;
    double* d_N;
    double* d_out;
    cudaMalloc(&d_M, sizeof(double) * m_size);
    cudaMalloc(&d_N, sizeof(double) * n_size);
    cudaMalloc(&d_out, sizeof(double) * output_size);
    cudaMemcpy(d_M, m.data(0,0), sizeof(double) * m_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, n.data(0,0), sizeof(double) * n_size, cudaMemcpyHostToDevice);
    dim3 threadDim{static_cast<unsigned int>(m_x), static_cast<unsigned int>(n_y), 1};
    from_cuda<<<1, threadDim>>>(d_M, d_N, d_out, m_y, n_y);
    cudaDeviceSynchronize();
    cudaMemcpy(result, d_out, sizeof(double) * output_size, cudaMemcpyDeviceToHost);
    cudaFree(d_M);
    cudaFree(d_N);
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

PYBIND11_MODULE(gpu_op, m){
    m.doc() = "matmul & conv with gpu";
    m.def("gpu_matmul_base", &gpu_matmul_base, "multiply 2 matrix");
}