#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernel.cuh>
#include <define.cuh>

namespace histogram 
{
    int get_FLOPs(const int size)
    {
        return size;
    }

    int get_bytes_transferred(const int size, const int bin_size)
    {
        return size * sizeof(float) + 2 * bin_size * sizeof(int);
    }

    void get_kernel_launch_params(const int size, const int bin_size, const unsigned int version, int& num_threads, int& threads_per_block, int& shared_mem_bytes)
    {
        threads_per_block = 512;
        if (0 == version)
        {
            num_threads = size / 128;
            shared_mem_bytes = 0;
        }
        return;
    }

    __device__ __forceinline__ int compute_bin_id(const float val, const int bin_size, const float lower_level, const float upper_level)
    {
        return val / (upper_level - lower_level) * bin_size;
    }

    __global__ void v0(float* data, int* bin, const int size, const int bin_size, const float lower_level, const float upper_level)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int stride = gridDim.x * blockDim.x * 4;
        while (idx < size - 3)
        {
            float4 reg = FETCH_FLOAT4(data[idx]);
            atomicAdd(&bin[compute_bin_id(reg.x, bin_size, lower_level, upper_level)], 1);
            atomicAdd(&bin[compute_bin_id(reg.y, bin_size, lower_level, upper_level)], 1);
            atomicAdd(&bin[compute_bin_id(reg.z, bin_size, lower_level, upper_level)], 1);
            atomicAdd(&bin[compute_bin_id(reg.w, bin_size, lower_level, upper_level)], 1);
            idx += stride;
        }
    }
}