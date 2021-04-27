//
// Created by 张梵 on 2021/4/25.
//

#ifndef MATMUL_CONV_MLU_OP_H
#define MATMUL_CONV_MLU_OP_H

#include <cstddef>
#include <cstdint>
#ifdef __cplusplus
extern "C" {
#endif

int MLUPowerDifferenceOp(float* input1,float* input2, int pow, float*output, int dims_a);
double* cpu_matmul(double* input, double* weight, std::size_t H, std::size_t K, std::size_t W);


#ifdef __cplusplus
}
#endif
#endif //MATMUL_CONV_MLU_OP_H
