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
            return size;
        return 0;
    }

    __global__ void reduce0(float* input, float* output, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size)   return;
        atomicAdd(output, input[idx]);
    }
}