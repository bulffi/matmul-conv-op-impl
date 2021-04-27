#include <cstdio>
#include "gpu_op.h"

__global__ void from_cuda() {
    printf("Hello from cuda: {%d, %d}\n", blockIdx.x, threadIdx.x);
}


int main() {
    from_cuda<<<1,10>>>();
    cudaDeviceSynchronize();
    return 0;
}