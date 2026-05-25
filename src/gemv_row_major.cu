#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernel.cuh>
#include <define.cuh>
#include <utils.cuh>
#include <types.cuh>

namespace gemv_row_major
{
    int get_FLOPs(const int rows, const int cols)
    {
        return 2 * rows * cols;
    }

    int get_bytes_transferred(const int rows, const int cols)
    {
        return rows * cols + cols + rows;
    }

    void get_kernel_launch_params(const int rows, const int cols, const unsigned int version, dim3& num_threads, dim3& threads_per_block, int& shared_mem_bytes)
    {
        if (0 == version)
        {
            threads_per_block = 512;
            num_threads = rows * threads_per_block.x;
            shared_mem_bytes = threads_per_block.x / 32 * sizeof(float);
        }
        else
        {
            threads_per_block = 0;
            num_threads = 0;
            shared_mem_bytes = 0;
        }
        return;
    }

    __global__ void v0(float* m, float* v, float* d_output, const int rows, const int cols)
    {
        AddOp<float> add_op;

        extern __shared__ float smem[];
        int row_idx = blockIdx.x;
        int row_stride = gridDim.x;
        while (row_idx < rows)
        {
            int col_idx = threadIdx.x * 4;
            int tid = threadIdx.x;
            float x = 0.f;
            int col_stride = blockDim.x * 4;
            while (col_idx < cols - 3)
            {
                float4 m_reg = FETCH_FLOAT4(m[row_idx * cols + col_idx]);
                float4 v_reg = FETCH_FLOAT4(v[col_idx]);
                x += m_reg.x * v_reg.x + m_reg.y * v_reg.y + m_reg.z * v_reg.z + m_reg.w * v_reg.w;
                col_idx += col_stride;
            }

            constexpr int warp_size = 32;
            x = shuffle_warp_reduce<warp_size, float, AddOp>(x, add_op);
            if (0 == (tid & (warp_size - 1)))
                smem[tid >> 5] = x;
            __syncthreads();

            constexpr int mini_warp_size = 16;
            if (tid < mini_warp_size)
                x = shuffle_warp_reduce<mini_warp_size, float, AddOp>(smem[tid], add_op);

            if (0 == tid)
                atomicAdd(d_output + row_idx, x);

            row_idx += row_stride;
        }
    }
}