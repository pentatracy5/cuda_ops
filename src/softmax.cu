#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernel.cuh>
#include <define.cuh>
#include <utils.cuh>
#include <types.cuh>

namespace softmax
{
    long long get_FLOPs(const long long rows, const long long cols)
    {
        return rows * ((cols - 1) * 2 + 4 * cols); // 对于每一行，reduce max/sum 均是 cols - 1 FLOPs，expf 和 / 粗略认为是 4 * cols FLOPs
    }

    long long get_bytes_transferred(const long long rows, const long long cols)
    {
        return rows * cols * sizeof(float) * 2;
    }

    void get_kernel_launch_params(const int rows, const int cols, const unsigned int version, dim3& num_threads, dim3& threads_per_block, int& shared_mem_bytes)
    {
        threads_per_block = 512;
        if (0 == version)
        {
            num_threads = (rows + 255) / 256 * threads_per_block.x;
            shared_mem_bytes = (threads_per_block.x / 32 + 1) * sizeof(float); // threads_per_block / 32 个 fp32 用于 reduce max/sum，1 个 fp32 用于存储 reduce max/sum 结果
        }
        else
        {
            num_threads = 0;
            shared_mem_bytes = 0;
        }
        return;
    }

    __global__ void v0(float* d_input, float* d_output, const int rows, const int cols)
    {
        MaxOp<float> max_op;
        AddOp<float> add_op;

        extern __shared__ float smem[];
        float& smem_reduce_res = smem[blockDim.x / 16];

        int row_stride = gridDim.x;
        int col_stride = blockDim.x * 4;
        int tid = threadIdx.x;
        int row_idx = blockIdx.x;
        int col_idx;
        while (row_idx < rows)
        {
            col_idx = threadIdx.x * 4;
            float row_reduce_res = FLT_MIN;
            while (col_idx < cols - 3)
            {
                float4 input = FETCH_FLOAT4(d_input[row_idx * cols + col_idx]);
                row_reduce_res = max(max(max(max(input.x, input.y), input.z), input.w), row_reduce_res);
                col_idx += col_stride;
            }

            constexpr int warp_size = 32;
            row_reduce_res = shuffle_warp_reduce<warp_size, float, MaxOp>(row_reduce_res, max_op);
            if (0 == (tid & (warp_size - 1)))
                smem[tid >> 5] = row_reduce_res;
            __syncthreads();

            constexpr int mini_warp_size = 16;
            if (tid < mini_warp_size)
                row_reduce_res = shuffle_warp_reduce<mini_warp_size, float, MaxOp>(smem[tid], max_op);

            if (0 == tid)
                smem_reduce_res = row_reduce_res;
            __syncthreads();
            float row_max = smem_reduce_res;

            col_idx = threadIdx.x * 4;
            row_reduce_res = 0.0f;
            while (col_idx < cols - 3)
            {
                float4 input = FETCH_FLOAT4(d_input[row_idx * cols + col_idx]);
                row_reduce_res = expf(input.x - row_max) + expf(input.y - row_max) + expf(input.z - row_max) + expf(input.w - row_max);
                col_idx += col_stride;
            }

            row_reduce_res = shuffle_warp_reduce<warp_size, float, AddOp>(row_reduce_res, add_op);
            if (0 == (tid & (warp_size - 1)))
                smem[tid >> 5] = row_reduce_res;
            __syncthreads();

            if (tid < mini_warp_size)
                row_reduce_res = shuffle_warp_reduce<mini_warp_size, float, AddOp>(smem[tid], add_op);

            if (0 == tid)
                smem_reduce_res = row_reduce_res;
            __syncthreads();

            col_idx = threadIdx.x * 4;
            while (col_idx < cols - 3)
            {
                float4 input = FETCH_FLOAT4(d_input[row_idx * cols + col_idx]);
                float4 output;
                output.x = expf(input.x - row_max) / smem_reduce_res;
                output.y = expf(input.y - row_max) / smem_reduce_res;
                output.z = expf(input.z - row_max) / smem_reduce_res;
                output.w = expf(input.w - row_max) / smem_reduce_res;
                FETCH_FLOAT4(d_output[row_idx * cols + col_idx]) = output;
                col_idx += col_stride;
            }

            row_idx += row_stride;
        }
    }
}