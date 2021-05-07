#include <cstdio>
#include <iostream>
#include <chrono>
#include "gpu_op.h"

__global__ void single_element(const double* M, const double* N, double* out, unsigned long size, unsigned long n_y) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        sum += M[threadIdx.x * size + i] * N[i * n_y + threadIdx.y];
    }
    out[threadIdx.x * n_y + threadIdx.y] = sum;
}

__global__ void block_element(const double* M, const double* N, double* out, unsigned long m_x, unsigned long m_y,
                              unsigned long n_y, unsigned long x_range, unsigned long y_range) {
    int x_start = threadIdx.x * x_range;
    int y_start = threadIdx.y * y_range;
    int x_end = x_start + x_range;
    int y_end = y_start + y_range;
    if (threadIdx.x == blockDim.x - 1) {
        x_end = m_x;
    }
    if (threadIdx.y == blockDim.y - 1) {
        y_end = n_y;
    }
    for (int i = x_start; i < x_end; i++) {
        for (int j = y_start; j < y_end; j++) {
            double tempt_sum = 0;
            for (int k = 0; k < m_y; k++) {
                tempt_sum += M[i * m_y + k] * N[k * n_y + j];
            }
            out[i * n_y + j] = tempt_sum;
        }
    }
}

__global__ void TB_aware_block_element(const double* M, const double* N, double* out, int m_x, int m_y,
                                       int n_y, int tb_x_range, int tb_y_range, int x_range, int y_range) {
    // first job is to understand x/y_start/end
    // the corner block can only be bigger, so the start address is always the same
    int x_start = blockIdx.x * tb_x_range + threadIdx.x * x_range;
    int y_start = blockIdx.y * tb_y_range + threadIdx.y * y_range;
    // end address requires additional attention
    int x_end = x_start + x_range;
    int y_end = y_start + y_range;

    if (blockIdx.x == gridDim.x - 1) {
        if (threadIdx.x == blockDim.x - 1) {
            x_end = m_x;
        }
    } else {
        if (threadIdx.x == blockDim.x - 1) {
            x_end = (blockIdx.x + 1) * tb_x_range;
        }
    }

    if (blockIdx.y == gridDim.y - 1) {
        if (threadIdx.y == blockDim.y - 1) {
            y_end = n_y;
        }
    } else {
        if (threadIdx.y == blockDim.y - 1) {
            y_end = (blockIdx.y + 1) * tb_y_range;
        }
    }

    for (int i = x_start; i < x_end; i++) {
        for (int j = y_start; j < y_end; j++) {
            double tempt_sum = 0;
            for (int k = 0; k < m_y; k++) {
                tempt_sum += M[i * m_y + k] * N[k * n_y + j];
            }
            out[i * n_y + j] = tempt_sum;
        }
    }
}

__global__ void TB_aware_block_element_tiling(const double* M, const double* N, double* out, int m_x, int m_y,
                                              int n_y) {
    __shared__ double M_shared[16][16];
    __shared__ double N_shared[16][16];
    unsigned point_x = blockIdx.x * 16 + threadIdx.x;
    unsigned point_y = blockIdx.y * 16 + threadIdx.y;

    double value = 0;
    for (int i = 0; i <= m_y / 16; i++) {
        if (i * 16 + threadIdx.y < m_y) {
            M_shared[threadIdx.x][threadIdx.y] = M[point_x * m_y + i * 16 + threadIdx.y];
        } else {
            M_shared[threadIdx.x][threadIdx.y] = 0;
        }
        if (i * 16 + threadIdx.x < m_y) {
            N_shared[threadIdx.x][threadIdx.y] = N[(i * 16 + threadIdx.x) * n_y + point_y];
        } else {
            N_shared[threadIdx.x][threadIdx.y] = 0;
        }
        __syncthreads();
        for (int j = 0; j < 16; j++) {
            value += M_shared[threadIdx.x][j] * N_shared[j][threadIdx.y];
        }
        __syncthreads();
    }
    if (point_x < m_x && point_y < n_y) {
        out[point_x * n_y + point_y] = value;
    }
}

pybind11::array_t<double> gpu_matmul_base(pybind11::array_t<double> M, pybind11::array_t<double> N) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
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
    // in this initial implementation, we try to use as many threads as we can!
    // in Turing architecture, we can have 1024(32*32) threads per block
    // so we divide it like this.
    double* d_M;
    double* d_N;
    double* d_out;
    cudaMalloc(&d_M, sizeof(double) * m_size);
    cudaMalloc(&d_N, sizeof(double) * n_size);
    cudaMalloc(&d_out, sizeof(double) * output_size);
    cudaMemcpy(d_M, m.data(0,0), sizeof(double) * m_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, n.data(0,0), sizeof(double) * n_size, cudaMemcpyHostToDevice);
    std::chrono::steady_clock::time_point real_begin = std::chrono::steady_clock::now();
    if (m_x <= 32 && n_y <= 32) {
        dim3 threadDim{static_cast<unsigned int>(m_x), static_cast<unsigned int>(n_y), 1};
        single_element<<<1, threadDim>>>(d_M, d_N, d_out, m_y, n_y);
    } else {
        std::size_t x_range = m_x / 32;
        std::size_t y_range = n_y / 32;
        dim3 threadDim{
            static_cast<unsigned int>(std::min((unsigned long)32, m_x)),
            static_cast<unsigned int>(std::min((unsigned long)32, n_y)),
            1
        };
        block_element<<<1, threadDim>>>(d_M, d_N, d_out, m_x, m_y, n_y, x_range, y_range);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(result, d_out, sizeof(double) * output_size, cudaMemcpyDeviceToHost);
    cudaFree(d_M);
    cudaFree(d_N);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    std::cout << "Real Time " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end - real_begin).count() << std::endl;
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

pybind11::array_t<double> gpu_matmul_multi_sm(pybind11::array_t<double> M, pybind11::array_t<double> N) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
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
    // in this initial implementation, we try to use as many thread blocks as we can
    // There are 30 Stream Multiprocessors in my Nvidia GTX 2060 graphics card
    // Each SM contains 64 CUDA cores
    // We try to divide the computation into 30 parts, then each parts can be further divided into threads
    // we use a 6*6 partition
    double* d_M;
    double* d_N;
    double* d_out;
    cudaMalloc(&d_M, sizeof(double) * m_size);
    cudaMalloc(&d_N, sizeof(double) * n_size);
    cudaMalloc(&d_out, sizeof(double) * output_size);
    cudaMemcpy(d_M, m.data(0,0), sizeof(double) * m_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, n.data(0,0), sizeof(double) * n_size, cudaMemcpyHostToDevice);
    std::chrono::steady_clock::time_point real_begin = std::chrono::steady_clock::now();
    if (m_x < 6 || n_y < 6) {
        dim3 threadDim{static_cast<unsigned int>(m_x), static_cast<unsigned int>(n_y), 1};
        single_element<<<1, threadDim>>>(d_M, d_N, d_out, m_y, n_y);
    } else {
        // we divide TB with 6*6
        // divide thread with 32*32 to try to use them ALL!
        int tb_x_range = m_x / 32;
        int tb_y_range = n_y / 32;
        int x_range = tb_x_range / 32;
        int y_range = tb_y_range / 32;
        dim3 blockDim{32, 32,1};
        dim3 threadDim{32, 32, 1};
        TB_aware_block_element<<<blockDim, threadDim>>>(d_M, d_N, d_out, m_x, m_y, n_y, tb_x_range, tb_y_range, x_range, y_range);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(result, d_out, sizeof(double) * output_size, cudaMemcpyDeviceToHost);
    cudaFree(d_M);
    cudaFree(d_N);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    std::cout << "Real Time " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end - real_begin).count() << std::endl;
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

pybind11::array_t<double> gpu_matmul_multi_sm_tiling(pybind11::array_t<double> M, pybind11::array_t<double> N) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
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
    // in this initial implementation, we try to use as many thread blocks as we can
    // There are 30 Stream Multiprocessors in my Nvidia GTX 2060 graphics card
    // Each SM contains 64 CUDA cores
    // We try to divide the computation into 30 parts, then each parts can be further divided into threads
    // we use a 6*6 partition
    double* d_M;
    double* d_N;
    double* d_out;
    cudaMalloc(&d_M, sizeof(double) * m_size);
    cudaMalloc(&d_N, sizeof(double) * n_size);
    cudaMalloc(&d_out, sizeof(double) * output_size);
    cudaMemcpy(d_M, m.data(0,0), sizeof(double) * m_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, n.data(0,0), sizeof(double) * n_size, cudaMemcpyHostToDevice);
    std::chrono::steady_clock::time_point real_begin = std::chrono::steady_clock::now();
    if (m_x < 16 || n_y < 16) {
        dim3 threadDim{static_cast<unsigned int>(m_x), static_cast<unsigned int>(n_y), 1};
        single_element<<<1, threadDim>>>(d_M, d_N, d_out, m_y, n_y);
    } else {
        // we simplify the process
        // we use 16*16 per thread block
        // each thread handle a single element
        int tb_x = m_x / 16 + 1;
        int tb_y = n_y / 16 + 1;
        dim3 grid_dim{static_cast<unsigned int>(tb_x), static_cast<unsigned int>(tb_y), 1};
        dim3 tb_dim{16, 16, 1};
        TB_aware_block_element_tiling<<<grid_dim, tb_dim>>>(d_M, d_N, d_out, m_x, m_y, n_y);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(result, d_out, sizeof(double) * output_size, cudaMemcpyDeviceToHost);
    cudaFree(d_M);
    cudaFree(d_N);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
    std::cout << "Real Time " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end - real_begin).count() << std::endl;
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
    m.def("gpu_matmul_multi_sm", &gpu_matmul_multi_sm, "multiply 2 matrix using many SMs");
    m.def("gpu_matmul_multi_sm_tiling", &gpu_matmul_multi_sm_tiling, "multiply 2 matrix using many SMs with"
                                                                     "each tiling");
}