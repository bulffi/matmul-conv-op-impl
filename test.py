import time

import cpu_op
import numpy as np
import gpu_op

M = np.random.rand(20, 900000)
N = np.random.rand(900000, 14)
# M = np.array([[1.0, 2.0], [3.0, 4.0]])
# N = np.array([[4.0, 5.0], [6.0, 7.0]])
R_ = np.matmul(M, N)

def cal_time(prefix, fn):
    t = time.process_time()
    res = fn(M, N)
    elapsed_time = time.process_time() - t
    print('{} time: {}'.format(prefix, elapsed_time))
    loss = np.sum((res - R_) ** 2)
    print('{} loss: {}'.format(prefix, loss))


cal_time('Numpy base', np.matmul)
cpu_op.cpu_matmul_base(M, N)
cal_time('CPU base', cpu_op.cpu_matmul_base)
cal_time('GPU base', gpu_op.gpu_matmul_base)
