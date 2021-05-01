//
// Created by 张梵 on 2021/4/25.
//

#include <iostream>
#include "cpu_op.h"
#include <chrono>
#include "thread_pool.h"

ThreadPool pool{16};

pybind11::array_t<double> cpu_matmul_base(const pybind11::array_t<double>& M, const pybind11::array_t<double>& N) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto m = M.unchecked<2>();
    auto n = N.unchecked<2>();
    std::size_t m_x = m.shape(0);
    std::size_t m_y = m.shape(1);
    std::size_t n_x = n.shape(0);
    std::size_t n_y = n.shape(1);
    std::size_t output_size = m_x * n_y;
    auto *result = new double[output_size];
    auto *m_addr = m.data(0, 0);
    auto *n_addr = n.data(0, 0);

    /// =========================================

    for (int i = 0; i < m_x; ++i) {
        for (int j = 0; j < n_y; ++j) {
            double tempt_sum = 0;
            for (int k = 0; k < m_y; ++k) {
                tempt_sum += m_addr[i * m_y + k] * n_addr[k * n_y + j];
            }
            result[i * n_y + j] = tempt_sum;
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
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

pybind11::array_t<double> cpu_matmul_transpose(const pybind11::array_t<double>& M, const pybind11::array_t<double>& N) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto m = M.unchecked<2>();
    auto n = N.unchecked<2>();
    std::size_t m_x = m.shape(0);
    std::size_t m_y = m.shape(1);
    std::size_t n_x = n.shape(0);
    std::size_t n_y = n.shape(1);
    std::size_t output_size = m_x * n_y;
    auto *result = new double[output_size];
    auto *m_addr = m.data(0, 0);
    auto *n_addr = n.data(0, 0);

    /// =========================================
//    we transpose the N matrix
    auto *new_n = new double [n_x * n_y];
    for (int i = 0; i < n_x; ++i) {
        for (int j = 0; j < n_y; ++j) {
            new_n[j * n_x + i] = n_addr[i * n_y + j];
        }
    }

    for (int i = 0; i < m_x; ++i) {
        for (int j = 0; j < n_y; ++j) {
            double tempt_sum = 0;
            for (int k = 0; k < m_y; ++k) {
                tempt_sum += m_addr[i * m_y + k] * new_n[j * n_x + k];
            }
            result[i * n.shape(1) + j] = tempt_sum;
        }
    }

    delete[] new_n;
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
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

pybind11::array_t<double> cpu_matmul_tiling(const pybind11::array_t<double>& M, const pybind11::array_t<double>& N) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto m = M.unchecked<2>();
    auto n = N.unchecked<2>();
    std::size_t m_x = m.shape(0);
    std::size_t m_y = m.shape(1);
    std::size_t n_x = n.shape(0);
    std::size_t n_y = n.shape(1);
    std::size_t output_size = m_x * n_y;
    auto *result = new double[output_size];
    auto *m_addr = m.data(0, 0);
    auto *n_addr = n.data(0, 0);

    /// =========================================
    // we understand that L1 cache line size is 64B
    // we can place 8 double in one cache line
    // we try to use them all!
    constexpr int TILE = 8;

    auto *m_end = m_addr + m_x * m_y;
    auto *n_end = n_addr + n_x * n_y;

    for (int i = 0; i < m_x; i += TILE) {
        for (int j = 0; j < n_y; j += TILE) {
            // now we partition the K
            for (int k = 0; k < m_y; k += TILE) {
                auto *m_base = m_addr + i * m_y + k;
                auto *n_base = n_addr + k * n_y + j;
                auto *out_base = result + i * n_y + j;
                // first we loop through rows of tile_m
                for (int l = 0; l < TILE && m_base + l * m_y < m_end; ++l) {
                    auto *m_row_base = m_base + l * m_y;
                    auto *m_row_end = m_addr + (i + l + 1) * m_y;
                    // then we loop through cols of tile_m
                    for (int i1 = 0; i1 < TILE && m_row_base + i1 < m_row_end; ++i1) {
                        auto *n_row_base = n_base + i1 * n_y;
                        auto *out_row_base = out_base + l * n_y;
                        auto *n_row_end = n_addr + (k + i1 + 1) * n_y;
                        // last we loop through cols of tile_n & tile_result
                        for (int k1 = 0; k1 < TILE && n_row_base + k1 < n_row_end; ++k1) {
                            out_row_base[k1] += m_row_base[i1] * n_row_base[k1];
                        }
                    }
                }
            }
        }
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
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

int multi_thread_worker(const double *M, const double *N, double *output, std::size_t m_x, std::size_t m_y,
                        std::size_t n_y, std::size_t start_i, std::size_t start_j, std::size_t stop_i, std::size_t stop_j) {

    // we understand that L1 cache line size is 64B
    // we can place 8 double in one cache line
    // we try to use them all!
    constexpr int TILE = 8;

    for (std::size_t i = start_i; i < stop_i; ++i) {
        for (std::size_t j = start_j; j < stop_j; ++j) {
            output[i * n_y + j] = 0;
        }
    }

    auto *m_end = M + stop_i * m_y;
    auto *n_end = N + m_y * n_y;

    for (std::size_t i = start_i; i < stop_i; i += TILE) {
        for (std::size_t j = start_j; j < stop_j; j += TILE) {
            // now we partition the K
            for (int k = 0; k < m_y; k += TILE) {
                auto *m_base = M + i * m_y + k;
                auto *n_base = N + k * n_y + j;
                auto *out_base = output + i * n_y + j;
                // first we loop through rows of tile_m
                for (int l = 0; l < TILE && m_base + l * m_y < m_end; ++l) {
                    auto *m_row_base = m_base + l * m_y;
                    auto *m_row_end = M + (i + l + 1) * m_y;
                    // then we loop through cols of tile_m
                    for (int i1 = 0; i1 < TILE && m_row_base + i1 < m_row_end; ++i1) {
                        auto *n_row_base = n_base + i1 * n_y;
                        auto *out_row_base = out_base + l * n_y;
                        auto *n_row_end = N + (k + i1) * n_y + stop_j;
                        // last we loop through cols of tile_n & tile_result
                        for (int k1 = 0; k1 < TILE && n_row_base + k1 < n_row_end; ++k1) {
//                            std::cout << m_row_base[i1] << " " << n_row_base[k1] << std::endl;
//                            std::cout << out_row_base[k1] << std::endl;
                            out_row_base[k1] += m_row_base[i1] * n_row_base[k1];
                        }
                    }
                }
            }
        }
    }
    return 0;
}

pybind11::array_t<double> cpu_matmul_multi_thread_tiling(const pybind11::array_t<double>& M, const pybind11::array_t<double>& N) {
    std::cout << "It's my turn" << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    auto m = M.unchecked<2>();
    auto n = N.unchecked<2>();
    std::size_t m_x = m.shape(0);
    std::size_t m_y = m.shape(1);
    std::size_t n_x = n.shape(0);
    std::size_t n_y = n.shape(1);
    std::size_t output_size = m_x * n_y;
    auto *result = new double[output_size];
    auto *m_addr = m.data(0, 0);
    auto *n_addr = n.data(0, 0);

    /// =========================================
    // this AMD core has 16 cores in it
    // we try to use them all
    // so we divide it into 4*4 squares
    std::size_t standard_x = m_x / 4;
    std::size_t standard_y = n_y / 4;
    std::vector<std::future<int>> waiter;
    // inside nodes
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            waiter.emplace_back(pool.enqueue(&multi_thread_worker, m_addr, n_addr, result, m_x, m_y,
                                             n_y, standard_x * i, standard_y * j,
                                             standard_x * (i + 1),standard_y * (j + 1)));

        }
    }
    // corner row
    for (int i = 0; i < 3; ++i) {
        waiter.emplace_back(pool.enqueue(&multi_thread_worker, m_addr, n_addr, result, m_x, m_y,
                                         n_y, standard_x * 3, standard_y * i, m_x,
                                         standard_y * (i + 1)));
    };

    // corner col
    for (int i = 0; i < 3; ++i) {
        waiter.emplace_back(pool.enqueue(&multi_thread_worker, m_addr, n_addr, result, m_x, m_y,
                                         n_y, standard_x * i, standard_x * 3,
                                         standard_y * (i + 1), n_y));
    }
    // corner point
    waiter.emplace_back(pool.enqueue(&multi_thread_worker, m_addr, n_addr, result, m_x, m_y,
                                     n_y, standard_x * 3, standard_y * 3, m_x, n_y));

    for (int i = 0; i < 16; ++i) {
        waiter[i].get();
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Time " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
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
    m.def("cpu_matmul_base", &cpu_matmul_base, "multiply 2 matrix with cpu");
    m.def("cpu_matmul_transpose", &cpu_matmul_transpose, "multiply with cpu transpose");
    m.def("cpu_matmul_tiling", &cpu_matmul_tiling, "multiply with cpu tiling");
    m.def("cpu_matmul_multi_thread_tiling", &cpu_matmul_multi_thread_tiling, "multiply with cpu "
                                                                             "16 threads & tiling");
}
