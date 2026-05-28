#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernel.cuh>
#include <define.cuh>
#include <curand_kernel.h>

namespace elementwise_dropout
{
    long long get_FLOPs(const long long size)
    {
        return 64 * size; // not accurate 
    }

    long long get_bytes_transferred(const long long size, const long long dir_vec_dim)
    {
        if constexpr (RANDTYPE == QUASI)
            return size * 2 * sizeof(float) + min(size, dir_vec_dim) * (sizeof(curandDirectionVectors32_t) + sizeof(unsigned int));
        else
            return size * 2 * sizeof(float) + sizeof(float);
    }

    void get_kernel_launch_params(const int size, const unsigned int version, dim3& num_threads, dim3& threads_per_block)
    {
        threads_per_block = 512;
        if (0 == version)
            num_threads = (size / 4 + 31) / 32;
        else if((sizeof(kernels) / sizeof(kernels[0]) - 1) == version)
            num_threads = size;
        return;
    }

    template <RandType rtype>
    __global__ void v0(float* input, float* output, const float p, curandDirectionVectors32_t* dir_vecs, unsigned int* scramble_constants, const int size, const int dir_vec_dim, const float* seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = gridDim.x * blockDim.x * 4;
        float scale = 1.0f / (1.0f - p);
        if constexpr (rtype == QUASI)
        {
            int vector_idx = int(float(idx) / stride * dir_vec_dim);
            curandStateScrambledSobol32_t states;
            curand_init(dir_vecs[vector_idx], scramble_constants[vector_idx], idx, &states);
            idx *= 4;
            while (idx < size - 3)
            {
                float4 reg = FETCH_FLOAT4(input[idx]);
                reg.x = curand_uniform(&states) < p ? 0.0f : reg.x * scale;
                reg.y = curand_uniform(&states) < p ? 0.0f : reg.y * scale;
                reg.z = curand_uniform(&states) < p ? 0.0f : reg.z * scale;
                reg.w = curand_uniform(&states) < p ? 0.0f : reg.w * scale;
                FETCH_FLOAT4(output[idx]) = reg;
                idx += stride;
            }
            while (idx < size)
            {
                output[idx] = curand_uniform(&states) < p ? 0.0f : input[idx] * scale;
                idx += 1;
            }
        }
        else
        {
            curandStatePhilox4_32_10_t states;
            curand_init(*seed * ULLONG_MAX, idx, 0, &states);
            idx *= 4;
            while (idx < size - 3)
            {
                float4 reg = FETCH_FLOAT4(input[idx]);
                float4 sample = curand_uniform4(&states);
                reg.x = sample.x < p ? 0.0f : reg.x * scale;
                reg.y = sample.y < p ? 0.0f : reg.y * scale;
                reg.z = sample.z < p ? 0.0f : reg.z * scale;
                reg.w = sample.w < p ? 0.0f : reg.w * scale;
                FETCH_FLOAT4(output[idx]) = reg;
                idx += stride;
            }
            while (idx < size)
            {
                output[idx] = curand_uniform(&states) < p ? 0.0f : input[idx] * scale;
                idx += 1;
            }
        }
    }
    template __global__ void v0<RANDTYPE>(float*, float*, const float, curandDirectionVectors32_t*, unsigned int*, const int, const int, const float*);

    template <RandType rtype>
    __global__ void v_ref(float* input, float* output, const float p, curandDirectionVectors32_t* dir_vecs, unsigned int* scramble_constants, const int size, const int dir_vec_dim, const float* seed)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size) return;
        float scale = 1.0f / (1.0f - p);
        if constexpr (rtype == QUASI)
        {
            curandStateScrambledSobol32_t states;
            int stride = gridDim.x * blockDim.x * 4;
            int vector_idx = int(float(idx) / stride * dir_vec_dim);
            curand_init(dir_vecs[vector_idx], scramble_constants[vector_idx], idx, &states);
            output[idx] = curand_uniform(&states) < p ? 0.0f : input[idx] * scale;
        }
        else
        {
            curandStatePhilox4_32_10_t states;
            curand_init(*seed * ULLONG_MAX, idx, 0, &states);
            output[idx] = curand_uniform(&states) < p ? 0.0f : input[idx] * scale;
        }
    }
    template __global__ void v_ref<RANDTYPE>(float*, float*, const float, curandDirectionVectors32_t*, unsigned int*, const int, const int, const float*);

}