#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernel.cuh>
#include <define.cuh>

namespace elementwise_add 
{
    int get_FLOPs(const int size)
    {
        return size;
    }

    int get_bytes_transferred(const int size)
    {
        return 3 * size * sizeof(float);
    }

    void get_kernel_launch_params(const int size, const unsigned int version, int& num_threads, int& threads_per_block) 
    {
        threads_per_block = 512;
        if (0 == version)
            num_threads = size;
        else if (1 == version)
            num_threads = size / 2;
        else if (2 == version)
            num_threads = size / 4;
        else
            num_threads = 0;
        return;
    }

    __global__ void no_vectorize(float* a, float* b, float* c, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size)   return;
        c[idx] = a[idx] + b[idx];
    }

    __global__ void vectorize2(float* a, float* b, float* c, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 2;
        if (idx >= size - 1)   return;
        float2 reg_a = FETCH_FLOAT2(a[idx]);
        float2 reg_b = FETCH_FLOAT2(b[idx]);
        float2 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        FETCH_FLOAT2(c[idx]) = reg_c;
    }

    __global__ void vectorize4(float* a, float* b, float* c, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        if (idx >= size - 3)   return;
        float4 reg_a = FETCH_FLOAT4(a[idx]);
        float4 reg_b = FETCH_FLOAT4(b[idx]);
        float4 reg_c;
        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;
        FETCH_FLOAT4(c[idx]) = reg_c;
    }
}