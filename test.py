import time

import cpu_op
import numpy as np
import gpu_op


# print(M[0][9])
# M = np.array([[1.0, 2.0], [3.0, 4.0]])
# N = np.array([[4.0, 5.0], [6.0, 7.0]])


def cal_time(prefix, fn):
    M = np.random.rand(10001, 10002)
    N = np.random.rand(10002, 10003)
    R_ = np.matmul(M, N)
    t = time.process_time()
    res = fn(M, N)
    elapsed_time = time.process_time() - t
    print('{} time: {}'.format(prefix, elapsed_time))
    loss = res - R_
    for i in range(100):
        for j in range(100):
            if abs(loss[i][j]) > 0.001:
                print(i, j, R_[i][j], res[i][j])
    loss = np.sum((res - R_) ** 2)
    print('{} loss: {}'.format(prefix, loss))


cal_time('Numpy base', np.matmul)
# cpu_op.cpu_matmul_base(M, N)
# cal_time('CPU base', cpu_op.cpu_matmul_base)
# cal_time('CPU transpose', cpu_op.cpu_matmul_transpose)
# cal_time('CPU tiling', cpu_op.cpu_matmul_tiling)
# cal_time('CPU thread tiling', cpu_op.cpu_matmul_multi_thread_tiling)
# cal_time('GPU base', gpu_op.gpu_matmul_base)
cal_time('GPU SMs', gpu_op.gpu_matmul_multi_sm)

# import mlu_op
# R = mlu_op.mlu_matmul_base(M, N)
# R = mlu_op.mlu_matmul_base(M, N)
# cal_time('MLU base', mlu_op.mlu_matmul_base)
# print(R)
