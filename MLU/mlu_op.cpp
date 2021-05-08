#include "mlu_op.h"
#include <stdlib.h>
#include "cnrt.h"
#include "cnrt_data.h"
#include "stdio.h"
#include <chrono>
#include <iostream>

extern "C" {
  void mlu_matmul_kernel_base(half* input1, half* input2, half* output, int32_t H, int32_t K, int32_t W);
}
cnrtDev_t dev;
class global_init {
public:
  global_init() {
    cnrtInit(0);
    // cnrtDev_t dev;
    cnrtGetDeviceHandle(&dev, 0);
    cnrtSetCurrentDevice(dev);
  }
  ~global_init() {
    cnrtDestroy();
  }
};

global_init init;

int mlu_matmul_base(const float* input,const float* weight, float* output, std::size_t H, std::size_t K, std::size_t W) {
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  cnrtQueue_t pQueue;
  cnrtCreateQueue(&pQueue);
  cnrtDim3_t dim;
  dim.x = 1;
  dim.y = 1;
  dim.z = 1;
  float hardware_time = 0.0;
  cnrtNotifier_t event_start;
  cnrtNotifier_t event_end;
  cnrtCreateNotifier(&event_start);
  cnrtCreateNotifier(&event_end);
  cnrtFunctionType_t c = CNRT_FUNC_TYPE_BLOCK;

  //prepare data
  half* input1_half = (half*)malloc(H * K * sizeof(half));
  half* input2_half = (half*)malloc(K * W * sizeof(half));
  half* output_half = (half*)malloc(H * W * sizeof(half));

  cnrtConvertFloatToHalfArray(input1_half, input, H * K);
  cnrtConvertFloatToHalfArray(input2_half, weight, K * W);
  // cnrtConvertFloatToHalfArray(output_half, output,dims_a);
 
  half *mlu_input1,*mlu_input2, *mlu_output;
  if (CNRT_RET_SUCCESS != cnrtMalloc((void**)&mlu_input1, H * K * sizeof(half))) {
    printf("cnrtMalloc Failed!\n");
    exit(-1);
  }
  if (CNRT_RET_SUCCESS != cnrtMalloc((void**)&mlu_input2, K * W * sizeof(half))) {
    printf("cnrtMalloc Failed!\n");
    exit(-1);
  }
  if (CNRT_RET_SUCCESS != cnrtMalloc((void**)&mlu_output, H * W * sizeof(half))) {
    printf("cnrtMalloc output Failed!\n");
    exit(-1);
  }
  // copy input into device
  cnrtMemcpy(mlu_input1, input1_half, H * K * sizeof(half), CNRT_MEM_TRANS_DIR_HOST2DEV);
  cnrtMemcpy(mlu_input2, input2_half, K * W * sizeof(half), CNRT_MEM_TRANS_DIR_HOST2DEV);
 
  //kernel parameters
  cnrtKernelParamsBuffer_t params;
  cnrtGetKernelParamsBuffer(&params);
  cnrtKernelParamsBufferAddParam(params, &mlu_input1, sizeof(half*)); 
  cnrtKernelParamsBufferAddParam(params, &mlu_input2, sizeof(half*)); 
  cnrtKernelParamsBufferAddParam(params, &mlu_output, sizeof(half*)); 
  cnrtKernelParamsBufferAddParam(params, &H, sizeof(int)); 
  cnrtKernelParamsBufferAddParam(params, &K, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &W, sizeof(int)); 
  cnrtPlaceNotifier(event_start, pQueue);

  std::chrono::steady_clock::time_point real_begin = std::chrono::steady_clock::now();
  cnrtInvokeKernel_V2((void*)&mlu_matmul_kernel_base, dim, params, c, pQueue);
  

  if (CNRT_RET_SUCCESS != cnrtSyncQueue(pQueue))
  {
    printf("syncQueue Failed!\n");
    exit(-1);
  }
  cnrtPlaceNotifier(event_end, pQueue);
  
  //get output data
  cnrtMemcpy(output_half, mlu_output, H * W * sizeof(half), CNRT_MEM_TRANS_DIR_DEV2HOST);

  cnrtConvertHalfToFloatArray(output, output_half, H * W);

  //free data
  if (CNRT_RET_SUCCESS != cnrtFree(mlu_input1)) {
    printf("cnrtFree Failed!\n");
    exit(-1);
  }
  if (CNRT_RET_SUCCESS != cnrtFree(mlu_input2)) {
    printf("cnrtFree Failed!\n");
    exit(-1);
  }
  if (CNRT_RET_SUCCESS != cnrtFree(mlu_output)) {
    printf("cnrtFree output Failed!\n");
    exit(-1);
  }
  if (CNRT_RET_SUCCESS != cnrtDestroyQueue(pQueue)) {
    printf("cnrtDestroyQueue Failed!\n");
    exit(-1);
  }
  if (CNRT_RET_SUCCESS != cnrtDestroyKernelParamsBuffer(params)) {
    printf("cnrtDestroyKernelParamsBuffer Failed!\n");
    return -1;
  }
  // cnrtDestroy();
  free(input1_half);
  free(input2_half);
  free(output_half);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
  std::cout << "Real Time " <<  std::chrono::duration_cast<std::chrono::milliseconds>(end - real_begin).count() << std::endl;

  return 0;
}