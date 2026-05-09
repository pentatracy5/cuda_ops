#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernel.cuh>
#include <define.cuh>

namespace elementwise_gelu 
{
    int get_FLOPs(const int size)
    {
        return 9 * size;
    }

    int get_bytes_transferred(const int size)
    {
        return 2 * size * sizeof(__half);
    }

    void get_kernel_launch_params(const int size, const unsigned int version, int& num_threads, int& threads_per_block) 
    {
        threads_per_block = 512;
        if (0 == version)
            num_threads = size / 128;
        return;
    }

    __global__ void v0(__half* input, __half* output, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 8;
        int stride = gridDim.x * blockDim.x * 8;
        while (idx < size - 7)
        {
            float4 reg_f_i = FETCH_FLOAT4(input[idx]);
            float4 reg_f_o;
            __half* reg_h_i = (__half*)&reg_f_i;
            __half* reg_h_o = (__half*)&reg_f_o;
            for (size_t i = 0; i < 8; i++)
                reg_h_o[i] = approximate_gelu(reg_h_i[i]);
            FETCH_FLOAT4(output[idx]) = reg_f_o;
            idx += stride;
        }
    }
}