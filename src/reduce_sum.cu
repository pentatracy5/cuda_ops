#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <kernel.cuh>
#include <define.cuh>

namespace reduce_sum 
{
    int get_FLOPs(const int size)
    {
        return size - 1;
    }

    int get_bytes_transferred(const int size)
    {
        return (size + 1) * sizeof(float);
    }

    int get_num_threads(const int size, const unsigned int version)
    {
        if (0 == version)
            return 1;
        if (1 == version)
            return size;
        return 0;
    }

    int get_shared_mem_size(const int threads_per_block, const unsigned int version)
    {
        if (0 == version)
            return 0;
        if (1 == version)
            return 0;
        if (2 == version)
            return threads_per_block;
        return 0;
    }

    __global__ void v0(float* input, float* output, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= 1)
            return;

        for (int i = 0; i < size; i++)
            output[0] += input[i];
    } 

    __global__ void v1(float* input, float* output, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size)
            return;

        atomicAdd(output, input[idx]);
    }
}