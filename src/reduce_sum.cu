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

    void get_kernel_launch_params(const int size, const unsigned int version, int& num_threads, int& threads_per_block, int& shared_mem_bytes)
    {
        threads_per_block = 1024;
        if (0 == version)
        {
            num_threads = 1;
            shared_mem_bytes = 0;
        }
        else if (1 == version)
        {
            num_threads = size;
            shared_mem_bytes = 0;
        }
        else if (2 == version)
        {
            num_threads = size;
            shared_mem_bytes = threads_per_block * sizeof(float);
        }
        else
        {
            num_threads = 0;
            shared_mem_bytes = 0;
        }
        return;
    }

    __global__ void v0(float* input, float* output, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= 1)
            return;
        float sum = 0.;
        for (int i = 0; i < size; i++)
            sum += input[i];
        output[0] += sum;
    } 

    __global__ void v1(float* input, float* output, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size)
            return;
        atomicAdd(output, input[idx]);
    }

    __global__ void v2(float* input, float* output, const int size)
    {
        extern __shared__ float mem[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int tid = threadIdx.x;
        int stride = blockDim.x;
        mem[tid] = idx >= size ? 0.f : input[idx];
        __syncthreads();
        while (stride > 1)
        {
            stride = stride >> 1;
            if (tid >= stride)
                return;
            mem[tid] += mem[tid + stride];
            __syncthreads();
        }
        atomicAdd(output, mem[0]);
    }
}