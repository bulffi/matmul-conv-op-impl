enable_language(CUDA)

pybind11_add_module(gpu_op gpu_op.cu)
set_target_properties(gpu_op PROPERTIES CUDA_ARCHITECTURES "75")