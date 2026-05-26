#pragma once

#include <cuda_runtime.h>

struct CudaTimer
{
    cudaEvent_t start, stop;
    CudaTimer();
    ~CudaTimer();
    void tic(cudaStream_t stream = 0);
    float toc(cudaStream_t stream = 0); // return milliseconds
};