//
// Created by 张梵 on 2021/4/25.
//

#ifndef MATMUL_CONV_GPU_OP_H
#define MATMUL_CONV_GPU_OP_H

#include <cstddef>

double* cpu_matmul(double* input, double* weight, std::size_t H, std::size_t K, std::size_t W);
double* cpu_conv2d(double* input, double* weight, std::size_t batch, std::size_t H, std::size_t W, \
                   std::size_t C, std::size_t w_H, std::size_t w_W, std::size_t w_batch);

#endif //MATMUL_CONV_GPU_OP_H
