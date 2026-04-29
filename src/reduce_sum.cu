#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <kernel.cuh>
#include <define.cuh>

namespace reduce_sum 
{
    int get_FLOPs(const int size)
    {
        return size - 1;
    }

    int get_bytes_transferred(const int size)
    {
        return (size + 1) * sizeof(float);
    }

    void get_kernel_launch_params(const int size, const unsigned int version, int& num_threads, int& threads_per_block, int& shared_mem_bytes)
    {
        threads_per_block = 512;
        if (0 == version)
        {
            num_threads = 1;
            shared_mem_bytes = 0;
        }
        else if (1 == version)
        {
            num_threads = size;
            shared_mem_bytes = 0;
        }
        else if (2 == version)
        {
            num_threads = size;
            shared_mem_bytes = threads_per_block * sizeof(float);
        }
        else if (3 == version)
        {
            num_threads = size / 4;
            shared_mem_bytes = threads_per_block * sizeof(float);
        }
        else if (4 == version)
        {
            num_threads = size / 4;
            shared_mem_bytes = threads_per_block * sizeof(float);
        }
        else
        {
            num_threads = 0;
            shared_mem_bytes = 0;
        }
        return;
    }

    /**
     * @brief 最朴素的并行归约求和内核（仅使用单个线程进行全局累加）
     *
     * @note 存在的开销与缺陷：
     *       - 单线程执行，性能较慢。
     *       - 当 size 非常大时，累加后期可能会出现“大数吞小数”现象，
     *         导致求和结果的相对误差显著增大。
     */
    __global__ void v0(float* input, float* output, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= 1)
            return;
        float sum = 0.;
        for (int i = 0; i < size; i++)
            sum += input[i];
        output[0] += sum;
    } 

    /**
     * @brief 基于原子操作的并行求和内核（每个线程直接累加到全局输出变量）
     *
     * @note 相比 v0 的改进：
     *       - 并行化内存访问，所有线程同时读取不同的数组元素，显著提升带宽利用率。
     *       
     * @note 存在的开销与缺陷：
     *       - 原子操作会使大量线程同时竞争同一内存地址（output[0]，导致严重的
     *         内存访问串行化，原子操作本身也有较高的硬件开销，当 size 很大时性能
     *         会显著下降（表现为大量线程阻塞等待）。
     *       - 浮点精度缺陷依然存在。
     */
    __global__ void v1(float* input, float* output, const int size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= size)
            return;
        atomicAdd(output, input[idx]);
    }

    /**
     * @brief 基于共享内存树形归约的并行求和内核（每个 block 内部并行归约）
     *
     * @note 相比 v1 的改进：
     *       - 使用共享内存进行 block 内并行归约，减少对全局内存的写入压力。
     *       - 每个 block 仅执行一次 atomicAdd，避免了大量线程同时竞争同一全局地址，
     *         大幅降低原子操作冲突，提升整体性能。
     *       - 在 block 内部，树形归约能够在一定程度上缓解“大数吞小数”问题。
     *       
     * @note 存在的开销与缺陷：
     *       - 当 block 数量很多时（即输入规模超大），最终需要执行多次 atomicAdd
     *         累加各个 block 的部分和。这些 atomicAdd 操作是串行化执行的，会降低性能。
     *         如果各个 block 的部分和之间量级差异悬殊，
     *       - 当 block 数量很多时（即输入规模超大），“大数吞小数”的问题将重新
     *         在全局累加阶段暴露出来。
     */
    __global__ void v2(float* input, float* output, const int size)
    {
        extern __shared__ float mem[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int tid = threadIdx.x;
        mem[tid] = idx >= size ? 0.f : input[idx];
        __syncthreads();
        int stride = blockDim.x;
        while (stride > 1)
        {
            stride = stride >> 1;
            if (tid >= stride)
                return;
            mem[tid] += mem[tid + stride];
            __syncthreads();
        }
        atomicAdd(output, mem[0]);
    }

    /**
     * @brief 基于向量化加载和共享内存树形归约的并行求和内核（每个线程一次处理4个元素）
     *
     * @note 相比 v2 的改进：
     *       - 使用 float4 向量化内存访问，每个线程一次读取4个连续浮点数，
     *         显著提升全局内存带宽利用率和合并访问效率。
     *       - 每个线程在归约开始前分别累加自己负责的4个元素，相当于在寄存器层面
     *         预先进行了局部归约，对于相同规模的问题，可以减少block数量，
     *         从而减少block切换的开销。
     *
     * @note 存在的开销与缺陷：
     *       - 当 block 数量很多时（即输入规模超大），原子操作的开销仍然存在。
     *       - 当 block 数量很多时（即输入规模超大），误差问题仍然存在。
     */
    __global__ void v3(float* input, float* output, const int size)
    {
        extern __shared__ float mem[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int tid = threadIdx.x;
        mem[tid] = 0.f;
        if (idx < size - 3)
        {
            float4 reg = FETCH_FLOAT4(input[idx]);
            mem[tid] += reg.x;
            mem[tid] += reg.y;
            mem[tid] += reg.z;
            mem[tid] += reg.w;
        }
        __syncthreads();
        int stride = blockDim.x;
        while (stride > 1)
        {
            stride = stride >> 1;
            if (tid >= stride)
                return;
            mem[tid] += mem[tid + stride];
            __syncthreads();
        }
        atomicAdd(output, mem[0]);
    }

    __device__ void warp_reduce(volatile float* cache, int tid)
    {
        cache[tid] += cache[tid + 32];
        cache[tid] += cache[tid + 16];
        cache[tid] += cache[tid + 8];
        cache[tid] += cache[tid + 4];
        cache[tid] += cache[tid + 2];
        cache[tid] += cache[tid + 1];
    }

    __global__ void v4(float* input, float* output, const int size)
    {
        extern __shared__ float mem[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int tid = threadIdx.x;
        mem[tid] = 0.f;
        if (idx < size - 3)
        {
            float4 reg = FETCH_FLOAT4(input[idx]);
            mem[tid] += reg.x;
            mem[tid] += reg.y;
            mem[tid] += reg.z;
            mem[tid] += reg.w;
        }
        __syncthreads();
        int stride = blockDim.x;
        while (stride > 64)
        {
            stride = stride >> 1;
            if (tid >= stride)
                return;
            mem[tid] += mem[tid + stride];
            __syncthreads();
        }
        if (tid >= 32)
            return;
        warp_reduce(mem, tid);
        if (tid == 0)
            atomicAdd(output, mem[0]);
    }
}