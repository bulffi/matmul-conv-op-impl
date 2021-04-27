//
// Created by 张梵 on 2021/4/25.
//

#ifndef MATMUL_CONV_MATMUL_H
#define MATMUL_CONV_MATMUL_H

#include <cstddef>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

pybind11::array_t<double> cpu_matmul(pybind11::array_t<double> M, pybind11::array_t<double> N);

#endif //MATMUL_CONV_MATMUL_H
