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

    int mlu_matmul(const float* input, const float* weight, float* output, std::size_t H, std::size_t K, std::size_t W);

#ifdef __cplusplus
}
#endif
#endif //MATMUL_CONV_MLU_OP_H
