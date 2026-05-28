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
        return size * 2 * sizeof(float) + min(size, dir_vec_dim) * (sizeof(curandDirectionVectors32_t) + sizeof(unsigned int));
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

    __global__ void v0(float* input, float* output, const float p, curandDirectionVectors32_t* dir_vecs, unsigned int* scramble_constants, const int size, const int dir_vec_dim)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = gridDim.x * blockDim.x * 4;
        int vector_idx = int(float(idx) / stride * dir_vec_dim);
        curandStateScrambledSobol32_t states;
        curand_init(dir_vecs[vector_idx], scramble_constants[vector_idx], idx, &states);
        idx *= 4;
        float scale = 1.0f / (1.0f - p);
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

    __global__ void v_ref(float* input, float* output, const float p, curandDirectionVectors32_t* dir_vecs, unsigned int* scramble_constants, const int size, const int dir_vec_dim)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size) return;
        curandStateScrambledSobol32_t states;
        int vector_idx = idx % dir_vec_dim;
        curand_init(dir_vecs[vector_idx], scramble_constants[vector_idx], idx, &states);
        float sample = curand_uniform(&states);
        output[idx] = sample < p ? 0.0f : input[idx] / (1.0f - p);
    }
}