#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernel.cuh>
#include <define.cuh>
#include <utils.cuh>
#include <types.cuh>

namespace gemv_col_major
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
            threads_per_block = 128;
            num_threads = rows;
            shared_mem_bytes = 0;
        }
        else if (1 == version)
        {
            threads_per_block = 64;
            num_threads = rows / 4;
            shared_mem_bytes = cols * sizeof(float);
        }
        else if (2 == version)
        {
            unsigned int num_blocks_y = 16;
            threads_per_block = 256;
            num_threads = dim3{ unsigned int(rows / 4), num_blocks_y };
            shared_mem_bytes = (cols / 4 + num_blocks_y - 1) / num_blocks_y * 4 * sizeof(float);
        }
        else
        {
            num_threads = 0;
            shared_mem_bytes = 0;
        }
        return;
    }

    __global__ void v0(float* m, float* v, float* d_output, const int rows, const int cols)
    {
        int row_idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (row_idx >= rows)
            return;
        float res = 0.0f;
        for (size_t col_idx = 0; col_idx < cols; col_idx++)
            res += m[col_idx * rows + row_idx] * v[col_idx];
        d_output[row_idx] = res;
    }

    __global__ void v1(float* m, float* v, float* d_output, const int rows, const int cols)
    {
        extern __shared__ float smem[];
        int col_idx = threadIdx.x * 4;
        int col_stride = blockDim.x * 4;
        while (col_idx < cols - 3)
        {
            FETCH_FLOAT4(smem[col_idx]) = FETCH_FLOAT4(v[col_idx]);
            col_idx += col_stride;
        }
        __syncthreads();

        int row_idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        int row_stride = gridDim.x * blockDim.x * 4;
        while (row_idx < rows - 3)
        {
            float4 res = { 0.0f, 0.0f, 0.0f, 0.0f };
            for (col_idx = 0; col_idx < cols; col_idx++)
            {
                float4 m_reg = FETCH_FLOAT4(m[col_idx * rows + row_idx]);
                float v_reg = smem[col_idx];
                res.x += m_reg.x * v_reg;
                res.y += m_reg.y * v_reg;
                res.z += m_reg.z * v_reg;
                res.w += m_reg.w * v_reg;
            }
            FETCH_FLOAT4(d_output[row_idx]) = res;
            row_idx += row_stride;
        }
    }

    __global__ void v2(float* m, float* v, float* d_output, const int rows, const int cols)
    {
        extern __shared__ float smem[];
        int col_range = (cols / 4 + gridDim.y - 1) / gridDim.y * 4;
        int col_start = blockIdx.y * col_range;
        int col_end = min(col_start + col_range, cols);
        col_range = col_end - col_start;

        int col_idx = threadIdx.x * 4;
        int col_stride = blockDim.x * 4;
        while (col_idx < col_range - 3)
        {
            FETCH_FLOAT4(smem[col_idx]) = FETCH_FLOAT4(v[col_idx + col_start]);
            col_idx += col_stride;
        }
        __syncthreads();

        int row_idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
        int row_stride = gridDim.x * blockDim.x * 4;
        while (row_idx < rows - 3)
        {
            float4 res = { 0.0f, 0.0f, 0.0f, 0.0f };
            for (col_idx = 0; col_idx < col_range; col_idx++)
            {
                float4 m_reg = FETCH_FLOAT4(m[(col_idx + col_start) * rows + row_idx]);
                float v_reg = smem[col_idx];
                res.x += m_reg.x * v_reg;
                res.y += m_reg.y * v_reg;
                res.z += m_reg.z * v_reg;
                res.w += m_reg.w * v_reg;
            }
            atomicAdd(d_output + row_idx + 0, res.x);
            atomicAdd(d_output + row_idx + 1, res.y);
            atomicAdd(d_output + row_idx + 2, res.z);
            atomicAdd(d_output + row_idx + 3, res.w);
            row_idx += row_stride;
        }
    }
}