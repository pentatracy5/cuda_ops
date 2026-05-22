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
        if (3 > version)
            num_threads = (size / 4 + 31) / 32;
        else if ((sizeof(kernels) / sizeof(kernels[0]) - 1) == version)
            num_threads = size;
        return;
    }

    __device__ __forceinline__ float fast_tanh(float x)
    {
        float r;
        asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
        return r;
    }

    __device__ __forceinline__ __half approximate_gelu_fast_tanh(__half x)
    {
        float x_f = __half2float(x);
        return __float2half_rn(x_f * 0.5f * (1.0f + fast_tanh(0.797884f * (x_f + 0.044715f * x_f * x_f * x_f))));
    }

    __global__ void v0(__half* input, __half* output, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 8;
        int stride = gridDim.x * blockDim.x * 8;
        while (idx < size - 7)
        {
            __half8 reg_h_i = FETCH_HALF8(input[idx]);
            __half8 reg_h_o;
            for (size_t i = 0; i < 8; i++)
                reg_h_o[i] = approximate_gelu_fast_tanh(reg_h_i[i]);
            FETCH_HALF8(output[idx]) = reg_h_o;
            idx += stride;
        }
    }

    __device__ __forceinline__ __half approximate_gelu_half(__half x)
    {
        const float tanh_in = __half2float(__float2half_rn(0.797884f) * (x + __float2half_rn(0.044715f) * x * x * x));
        const float tanh_out = fast_tanh(tanh_in);
        return __float2half_rn(0.5f) * x * (__float2half_rn(1.0f) + __float2half_rn(tanh_out));
    }

    __global__ void v1(__half* input, __half* output, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 8;
        int stride = gridDim.x * blockDim.x * 8;
        while (idx < size - 7)
        {
            __half8 reg_h_i = FETCH_HALF8(input[idx]);
            __half8 reg_h_o;
            for (size_t i = 0; i < 8; i++)
                reg_h_o[i] = approximate_gelu_half(reg_h_i[i]);
            FETCH_HALF8(output[idx]) = reg_h_o;
            idx += stride;
        }
    }

    __device__ __forceinline__ __half2 approximate_gelu_half2(__half2 x2)
    {
        const float2 tanh_in = __half22float2(__hmul2(__float2half2_rn(0.797884f), __hadd2(x2, __hmul2(__hmul2(__hmul2(__float2half2_rn(0.044715f), x2), x2), x2))));
        float2 tanh_out;
        tanh_out.x = fast_tanh(tanh_in.x);
        tanh_out.y = fast_tanh(tanh_in.y);
        return __hmul2(__hmul2(__float2half2_rn(0.5f), x2), __hadd2(__float2half2_rn(1.0f), __float22half2_rn(tanh_out)));
    }

    __global__ void v2(__half* input, __half* output, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 8;
        int stride = gridDim.x * blockDim.x * 8;
        while (idx < size - 7)
        {
            __half8 reg_h_i = FETCH_HALF8(input[idx]);
            __half8 reg_h_o;
            for (size_t i = 0; i < 8; i += 2)
                FETCH_HALF2(reg_h_o[i]) = approximate_gelu_half2(FETCH_HALF2(reg_h_i[i]));
            FETCH_HALF8(output[idx]) = reg_h_o;
            idx += stride;
        }
    }

    __device__ __forceinline__ __half approximate_gelu(__half x)
    {
        float x_f = __half2float(x);
        return __float2half_rn(x_f * 0.5f * (1.0f + tanhf(0.797884f * (x_f + 0.044715f * x_f * x_f * x_f))));
    }

    __global__ void v_ref(__half* input, __half* output, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            output[idx] = approximate_gelu(input[idx]);
    }
}