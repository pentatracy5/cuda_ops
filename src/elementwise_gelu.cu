#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernel.cuh>
#include <define.cuh>
#include <math_constants.h>

namespace elementwise_gelu 
{
    int get_FLOPs(const int size)
    {
        return 10 * size;
    }

    int get_bytes_transferred(const int size)
    {
        return 2 * size * sizeof(float);
    }

    void get_kernel_launch_params(const int size, const unsigned int version, int& num_threads, int& threads_per_block) 
    {
        threads_per_block = 512;
        if (0 == version)
            num_threads = size / 4;
        return;
    }

    __global__ void v0(float* input, float* output, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int stride = gridDim.x * blockDim.x * 4;
        while (idx < size - 3)
        {
            float4 reg_i = FETCH_FLOAT4(input[idx]);
            float4 reg_o;
            reg_o.x = reg_i.x * 0.5f * (1.0f + tanhf(0.797884f * (reg_i.x + 0.044715f * reg_i.x * reg_i.x * reg_i.x)));
            reg_o.y = reg_i.y * 0.5f * (1.0f + tanhf(0.797884f * (reg_i.y + 0.044715f * reg_i.y * reg_i.y * reg_i.y)));
            reg_o.z = reg_i.z * 0.5f * (1.0f + tanhf(0.797884f * (reg_i.z + 0.044715f * reg_i.z * reg_i.z * reg_i.z)));
            reg_o.w = reg_i.w * 0.5f * (1.0f + tanhf(0.797884f * (reg_i.w + 0.044715f * reg_i.w * reg_i.w * reg_i.w)));
            FETCH_FLOAT4(output[idx]) = reg_o;
            idx += stride;
        }
    }
}