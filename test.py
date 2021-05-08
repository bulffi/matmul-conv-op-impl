import time

import cpu_op
import numpy as np
import gpu_op


# print(M[0][9])
# M = np.array([[1.0, 2.0], [3.0, 4.0]])
# N = np.array([[4.0, 5.0], [6.0, 7.0]])


def cal_time(prefix, fn):
    M = np.random.rand(1024, 1024) * 0.1 - 0.05
    N = np.random.rand(1024, 1024) * 0.1 - 0.05
    # M = np.ones((1024, 1024))
    # N = np.ones((1024, 1024))
    R_ = np.matmul(M, N)
    t = time.time()
    res = fn(M, N)
    elapsed_time = time.time() - t
    print('{} time: {}'.format(prefix, elapsed_time))
    loss = res - R_
    for i in range(256):
        for j in range(256):
            if abs(loss[i][j]) > 0.1:
                print(i, j, R_[i][j], res[i][j])
    loss = np.sum((res - R_) ** 2)
    print('{} loss: {}'.format(prefix, loss))
    # breakpoint()



# cal_time('Numpy base', np.matmul)
# cpu_op.cpu_matmul_base(M, N)
# cal_time('CPU base', cpu_op.cpu_matmul_base)
# cal_time('CPU transpose', cpu_op.cpu_matmul_transpose)
# cal_time('CPU tiling', cpu_op.cpu_matmul_tiling)
cal_time('CPU thread tiling', cpu_op.cpu_matmul_multi_thread_tiling)
# cal_time('GPU base', gpu_op.gpu_matmul_base)
# cal_time('GPU SMs', gpu_op.gpu_matmul_multi_sm)
# cal_time('GPU SMs tiling', gpu_op.gpu_matmul_multi_sm_tiling)
import mlu_op
# R = mlu_op.mlu_matmul_base(M, N)
# R = mlu_op.mlu_matmul_base(M, N)
# cal_time('MLU base', mlu_op.mlu_matmul_base)
# cal_time('MLU multi core', mlu_op.mlu_matmul_multicore)
cal_time('MLU multi sram', mlu_op.mlu_matmul_multi_sram_wrapper)
# print(R)
