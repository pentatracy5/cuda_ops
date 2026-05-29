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

    long long get_bytes_transferred(const long long size, const dim3 num_threads)
    {
        return size * 2 * sizeof(float) + (num_threads.x * num_threads.y * num_threads.z) * 2 * sizeof(GetRandStateType<RANDTYPE>::Type);
    }

    void get_kernel_launch_params(const int size, const unsigned int version, dim3& num_threads, dim3& threads_per_block)
    {
        threads_per_block = 512;
        if (0 == version)
            num_threads = (size / 4 + 31) / 32;
        else if ((sizeof(kernels) / sizeof(kernels[0]) - 1) == version)
            num_threads = size;
        return;
    }

    template <RandType rtype>
    __global__ void setup_states(GetRandStateType<rtype>::Type* states, const int size, curandDirectionVectors32_t* dir_vecs, unsigned int* scramble_constants, const int seed, const int dir_vec_dim)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size) return;
        if constexpr (rtype == QUASI)
        {
            int vector_idx = int(float(idx) / size * dir_vec_dim);
            curand_init(dir_vecs[vector_idx], scramble_constants[vector_idx], idx, states + idx);
        }
        else
            curand_init(seed, idx, 0, states + idx);
    }
    template __global__ void setup_states<RANDTYPE>(GetRandStateType<RANDTYPE>::Type*, const int, curandDirectionVectors32_t*, unsigned int*, const int, const int);

    template <RandType rtype>
    __global__ void v0(float* input, float* output, const float p, GetRandStateType<rtype>::Type* states, const int size)
    {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        if (gid >= size) return;
        int idx = gid * 4;
        int stride = gridDim.x * blockDim.x * 4;
        float scale = 1.0f / (1.0f - p);
        if constexpr (rtype == QUASI)
        {
            typename GetRandStateType<rtype>::Type state = states[gid];
            while (idx < size - 3)
            {
                float4 reg = FETCH_FLOAT4(input[idx]);
                reg.x = float(curand_uniform(&state) >= p) * reg.x * scale;
                reg.y = float(curand_uniform(&state) >= p) * reg.y * scale;
                reg.z = float(curand_uniform(&state) >= p) * reg.z * scale;
                reg.w = float(curand_uniform(&state) >= p) * reg.w * scale;
                FETCH_FLOAT4(output[idx]) = reg;
                idx += stride;
            }
            while (idx < size)
            {
                output[idx] = float(curand_uniform(&state) >= p) * input[idx] * scale;
                idx += 1;
            }
            states[gid] = state;
        }
        else
        {
            typename GetRandStateType<rtype>::Type state = states[gid];
            while (idx < size - 3)
            {
                float4 reg = FETCH_FLOAT4(input[idx]);
                float4 sample = curand_uniform4(&state);
                reg.x = float(sample.x >= p) * reg.x * scale;
                reg.y = float(sample.y >= p) * reg.y * scale;
                reg.z = float(sample.z >= p) * reg.z * scale;
                reg.w = float(sample.w >= p) * reg.w * scale;
                FETCH_FLOAT4(output[idx]) = reg;
                idx += stride;
            }
            while (idx < size)
            {
                output[idx] = float(curand_uniform(&state) >= p) * input[idx] * scale;
                idx += 1;
            }
            states[gid] = state;
        }
    }
    template __global__ void v0<RANDTYPE>(float*, float*, const float, GetRandStateType<RANDTYPE>::Type*, const int);

    template <RandType rtype>
    __global__ void v_ref(float* input, float* output, const float p, GetRandStateType<rtype>::Type* states, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size) return;
        float scale = 1.0f / (1.0f - p);
        if constexpr (rtype == QUASI)
        {
            typename GetRandStateType<rtype>::Type state = states[idx];
            output[idx] = float(curand_uniform(&state) >= p) * input[idx] * scale;
            states[idx] = state;
        }
        else
        {
            typename GetRandStateType<rtype>::Type state = states[idx];
            output[idx] = float(curand_uniform(&state) >= p) * input[idx] * scale;
            states[idx] = state;
        }
    }
    template __global__ void v_ref<RANDTYPE>(float*, float*, const float, GetRandStateType<RANDTYPE>::Type*, const int);
}