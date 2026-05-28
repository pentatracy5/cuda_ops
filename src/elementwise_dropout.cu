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

    long long get_bytes_transferred(const long long size)
    {
        return size * (2 * sizeof(float) + sizeof(curandDirectionVectors32_t) + sizeof(unsigned int));
    }

    void get_kernel_launch_params(const int size, const unsigned int version, dim3& num_threads, dim3& threads_per_block)
    {
        threads_per_block = 512;
        if ((sizeof(kernels) / sizeof(kernels[0]) - 1) == version)
            num_threads = size;
        return;
    }

    __global__ void v_ref(float* input, float* output, const float p, curandDirectionVectors32_t* dir_vecs, unsigned int* scramble_constants, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size) return;
        curandStateScrambledSobol32_t states;
        int vector_idx = idx % 20000;
        curand_init(dir_vecs[vector_idx], scramble_constants[vector_idx], idx, &states);
        float sample = curand_uniform(&states);
        output[idx] = sample < p ? 0.0f : input[idx] / (1.0f - p);
    }
}