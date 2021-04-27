enable_language(CUDA)

pybind11_add_module(gpu_op gpu_op.cu)