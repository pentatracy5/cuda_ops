#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <kernel.cuh>
#include <define.cuh>
#include <utils.cuh>
#include <types.cuh>

namespace quantize
{
    int get_FLOPs(const int rows, const int cols)
    {
        return rows * ((cols - 1) * 2 + 6 + 4 * cols); // 对于每一行，reduce max/min 均是 cols - 1 FLOPs，计算 scale 和 zeropoint 粗略认为是 6 FLOPs，最后 fp32 转 int8 粗略认为是 4 * cols FLOPs
    }

    int get_bytes_transferred(const int rows, const int cols)
    {
        return rows * cols * (sizeof(float) + sizeof(int8_t));
    }

    void get_kernel_launch_params(const int rows, const int cols, const unsigned int version, int& num_threads, int& threads_per_block, int& shared_mem_bytes)
    {
        threads_per_block = 512;
        if (0 == version)
        {
            num_threads = (rows + 255) / 256 * threads_per_block;
            shared_mem_bytes = (threads_per_block / 16 + 2) * sizeof(float); // threads_per_block / 16 个 fp32 用于 reduce max/min，两个 fp32 用于存储 scale 和 zeropoint
        }
        else
        {
            num_threads = 0;
            shared_mem_bytes = 0;
        }
        return;
    }

    template<int warp_size>
    __device__ __forceinline__ float shuffle_warp_reduce_max(float x)
    {
        constexpr unsigned int mask = (1ULL << warp_size) - 1;
        if (32 <= warp_size)    x = max(x, __shfl_down_sync(mask, x, 16));
        if (16 <= warp_size)    x = max(x, __shfl_down_sync(mask, x, 8));
        if (8 <= warp_size)     x = max(x, __shfl_down_sync(mask, x, 4));
        if (4 <= warp_size)     x = max(x, __shfl_down_sync(mask, x, 2));
        if (2 <= warp_size)     x = max(x, __shfl_down_sync(mask, x, 1));
        return x;
    }

    template<int warp_size>
    __device__ __forceinline__ float shuffle_warp_reduce_min(float x)
    {
        constexpr unsigned int mask = (1ULL << warp_size) - 1;
        if (32 <= warp_size)    x = min(x, __shfl_down_sync(mask, x, 16));
        if (16 <= warp_size)    x = min(x, __shfl_down_sync(mask, x, 8));
        if (8 <= warp_size)     x = min(x, __shfl_down_sync(mask, x, 4));
        if (4 <= warp_size)     x = min(x, __shfl_down_sync(mask, x, 2));
        if (2 <= warp_size)     x = min(x, __shfl_down_sync(mask, x, 1));
        return x;
    }

    template <QuantizeType qtype>
    __global__ void v0(float* d_input, int8_t* d_output, const int rows, const int cols, const float qmin, const float qmax)
    {
        MaxOp<float> max_op;
        MinOp<float> min_op;

        extern __shared__ float smem[];
        float* smem_max = smem;
        float* smem_min = smem + blockDim.x / 32;
        float& smem_scale = smem[blockDim.x / 16];
        float& smem_zeropoint = smem[blockDim.x / 16 + 1];

        int row_stride = gridDim.x;
        int col_stride = blockDim.x * 4;
        int tid = threadIdx.x;
        int row_idx = blockIdx.x;
        int col_idx;
        while (row_idx < rows)
        {
            col_idx = threadIdx.x * 4;
            float row_max = FLT_MIN;
            float row_min = FLT_MAX;
            while (col_idx < cols - 3)
            {
                float4 input = FETCH_FLOAT4(d_input[row_idx * cols + col_idx]);
                row_max = max(max(max(max(input.x, input.y), input.z), input.w), row_max);
                row_min = min(min(min(min(input.x, input.y), input.z), input.w), row_min);
                col_idx += col_stride;
            }

            constexpr int warp_size = 32;
            row_max = shuffle_warp_reduce<warp_size, float, MaxOp>(row_max, max_op);
            row_min = shuffle_warp_reduce<warp_size, float, MinOp>(row_min, min_op);
            if (0 == (tid & (warp_size - 1)))
            {
                smem_max[tid >> 5] = row_max;
                smem_min[tid >> 5] = row_min;
            }
            __syncthreads();

            constexpr int mini_warp_size = 16;
            if (tid < mini_warp_size)
            {
                row_max = shuffle_warp_reduce<mini_warp_size, float, MaxOp>(smem_max[tid], max_op);
                row_min = shuffle_warp_reduce<mini_warp_size, float, MinOp>(smem_min[tid], min_op);
            }

            if (0 == tid)
            {
                if constexpr (qtype == ASYMMETRIC)
                {
                    smem_scale = (row_max - row_min) / (qmax - qmin);
                    smem_zeropoint = qmin - nearbyintf(row_min / smem_scale);
                }
                else
                {
                    smem_scale = max(fabs(row_max), fabs(row_min)) / qmax;
                    smem_zeropoint = 0.f;
                }
            }
            __syncthreads();

            col_idx = threadIdx.x * 4;
            while (col_idx < cols - 3)
            {
                float4 input = FETCH_FLOAT4(d_input[row_idx * cols + col_idx]);
                char4 output;
                output.x = clamp(nearbyintf(input.x / smem_scale + smem_zeropoint), qmin, qmax);
                output.y = clamp(nearbyintf(input.y / smem_scale + smem_zeropoint), qmin, qmax);
                output.z = clamp(nearbyintf(input.z / smem_scale + smem_zeropoint), qmin, qmax);
                output.w = clamp(nearbyintf(input.w / smem_scale + smem_zeropoint), qmin, qmax);
                FETCH_CHAR4(d_output[row_idx * cols + col_idx]) = output;
                col_idx += col_stride;
            }

            row_idx += row_stride;
        }
    }

    template __global__ void v0<QUANTIZETYPE>(float*, int8_t*, const int, const int, const float, const float);
}