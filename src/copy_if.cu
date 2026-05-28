#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernel.cuh>
#include <define.cuh>

namespace copy_if 
{
    long long get_FLOPs(const long long size)
    {
        return size;
    }

    long long get_bytes_transferred(const long long size)
    {
        return size * sizeof(float) + 2 * sizeof(int);
    }

    void get_kernel_launch_params(const int size, const unsigned int version, dim3& num_threads, dim3& threads_per_block, int& shared_mem_bytes)
    {
        threads_per_block = 512;
        if (2 > version)
        {
            num_threads = (size / 4 + 7) / 8;
            shared_mem_bytes = 0;
        }
        else if (2 <= version && 4 > version)
        {
            num_threads = (size / 4 + 7) / 8;
            shared_mem_bytes = sizeof(int);
        }
        return;
    }

    /**
     * @brief 基础版本的 copy_if 核函数（v0）
     * @details 每个线程一次处理 4 个元素（float4），直接使用全局原子操作 atomicAdd
     *          来分配输出位置。简单但原子竞争严重。
     */
    __global__ void v0(float* src, float* dst, int* dst_size, const int size, const float compare)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int stride = gridDim.x * blockDim.x * 4;
        while (idx < size - 3)
        {
            float4 reg = FETCH_FLOAT4(src[idx]);
            if (reg.x < compare)
                dst[atomicAdd(dst_size, 1)] = reg.x;
            if (reg.y < compare)
                dst[atomicAdd(dst_size, 1)] = reg.y;
            if (reg.z < compare)
                dst[atomicAdd(dst_size, 1)] = reg.z;
            if (reg.w < compare)
                dst[atomicAdd(dst_size, 1)] = reg.w;
            idx += stride;
        }
    }

    /**
     * @brief Warp 级原子聚合递增函数
     * @details 在 warp 内计算活跃线程数，仅让 leader 线程执行一次全局 atomicAdd，
     *          然后将结果通过 shuffle 广播，并根据 lane 内排名计算每个线程的局部偏移。
     *          可有效减少全局原子操作次数。
     * @param ctr 指向全局计数器的指针
     * @return 当前线程在本次聚合操作中的输出位置偏移
     */
    __device__ int atomic_aggregate_increment(int* ctr)
    {
        unsigned int active = __activemask();
        int leader = __ffs(active) - 1;
        int change = __popc(active);
        int lane_mask_lt;
        asm("mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt));
        unsigned int rank = __popc(active & lane_mask_lt);
        int warp_res;
        if (rank == 0)
            warp_res = atomicAdd(ctr, change);
        warp_res = __shfl_sync(active, warp_res, leader);
        return warp_res + rank;
    }

    /**
     * @brief 使用 warp 聚合原子操作的 copy_if 核函数（v1）
     * @details 与 v0 类似，但用 atomic_aggregate_increment 代替 atomicAdd，
     *          以 warp 为单位减少全局原子冲突。每次处理 4 个元素，并插入 __syncwarp
     *          保证数据一致性。
     * @note 由于 atomic_aggregate_increment 中调用了 __activemask()，为了保证拿到
     *       准确的 active，必须在 if (reg.x/y/z/w < compare) 的判断之前执行
     *       __syncwarp ，这样才能保证逻辑上命中 if (reg.x/y/z/w < compare) 分支的 thread
     *       能在实际运行时同步进入该分支
     */
    __global__ void v1(float* src, float* dst, int* dst_size, const int size, const float compare)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int stride = gridDim.x * blockDim.x * 4;
        while (idx < size - 3)
        {
            unsigned int mask = (unsigned int)((1ULL << min(32, (size - (idx & ~127)) >> 2)) - 1ULL);
            float4 reg = FETCH_FLOAT4(src[idx]);
            __syncwarp(mask);
            if (reg.x < compare)
                dst[atomic_aggregate_increment(dst_size)] = reg.x;
            __syncwarp(mask);
            if (reg.y < compare)
                dst[atomic_aggregate_increment(dst_size)] = reg.y;
            __syncwarp(mask);
            if (reg.z < compare)
                dst[atomic_aggregate_increment(dst_size)] = reg.z;
            __syncwarp(mask);
            if (reg.w < compare)
                dst[atomic_aggregate_increment(dst_size)] = reg.w;
            idx += stride;
        }
    }

    /**
     * @brief 使用共享内存进行块内局部重排的 copy_if 核函数（v2）
     * @details 每个线程块内先利用共享内存计数器 ssize 统计本块输出的元素数，
     *          再通过一次全局 atomicAdd 为整个块分配连续输出区间，
     *          最后各线程将符合条件的元素写入对应位置。避免了大量全局原子操作。
     *          共享内存大小需在启动时指定为 sizeof(int)。
     */
    __global__ void v2(float* src, float* dst, int* dst_size, const int size, const float compare)
    {
        extern __shared__ int ssize[];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int stride = gridDim.x * blockDim.x * 4;
        float4 reg;
        int4 pos; 
        while (true)
        {
            if (0 == tid && idx < size - 3)
                *ssize = 0;
            __syncthreads();

            if (idx < size - 3)
            {
                reg = FETCH_FLOAT4(src[idx]);
                if (reg.x < compare)
                    pos.x = atomicAdd(ssize, 1);
                if (reg.y < compare)
                    pos.y = atomicAdd(ssize, 1);
                if (reg.z < compare)
                    pos.z = atomicAdd(ssize, 1);
                if (reg.w < compare)
                    pos.w = atomicAdd(ssize, 1);
            }
            __syncthreads();

            if (0 == tid && idx < size - 3)
                *ssize = atomicAdd(dst_size, *ssize);
            __syncthreads();

            if (idx < size - 3)
            {
                if (reg.x < compare)
                    dst[pos.x + *ssize] = reg.x;
                if (reg.y < compare)
                    dst[pos.y + *ssize] = reg.y;
                if (reg.z < compare)
                    dst[pos.z + *ssize] = reg.z;
                if (reg.w < compare)
                    dst[pos.w + *ssize] = reg.w;
            }
            __syncthreads();

            if (idx >= size - 3)
                *ssize = -1;
            __syncthreads();

            if (*ssize == -1)
                return;

            idx += stride;
        }
    }

    /**
     * @brief 结合共享内存和 warp 聚合原子的 copy_if 核函数（v3）
     * @details 在 v2 的框架上，将块内统计的 atomicAdd 替换为 atomic_aggregate_increment，
     *          进一步减少块内原子冲突。结合了 warp 级优化和块级重排的优点。
     *          共享内存大小需在启动时指定为 sizeof(int)。
     */
    __global__ void v3(float* src, float* dst, int* dst_size, const int size, const float compare)
    {
        extern __shared__ int ssize[];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int stride = gridDim.x * blockDim.x * 4;
        float4 reg;
        int4 pos;
        while (true)
        {
            if (0 == tid && idx < size - 3)
                *ssize = 0;
            __syncthreads();

            if (idx < size - 3)
            {
                unsigned int mask = (unsigned int)((1ULL << min(32, (size - (idx & ~127)) >> 2)) - 1ULL);
                reg = FETCH_FLOAT4(src[idx]);
                __syncwarp(mask);
                if (reg.x < compare)
                    pos.x = atomic_aggregate_increment(ssize);
                __syncwarp(mask);
                if (reg.y < compare)
                    pos.y = atomic_aggregate_increment(ssize);
                __syncwarp(mask);
                if (reg.z < compare)
                    pos.z = atomic_aggregate_increment(ssize);
                __syncwarp(mask);
                if (reg.w < compare)
                    pos.w = atomic_aggregate_increment(ssize);
            }
            __syncthreads();

            if (0 == tid && idx < size - 3)
                *ssize = atomicAdd(dst_size, *ssize);
            __syncthreads();

            if (idx < size - 3)
            {
                if (reg.x < compare)
                    dst[pos.x + *ssize] = reg.x;
                if (reg.y < compare)
                    dst[pos.y + *ssize] = reg.y;
                if (reg.z < compare)
                    dst[pos.z + *ssize] = reg.z;
                if (reg.w < compare)
                    dst[pos.w + *ssize] = reg.w;
            }
            __syncthreads();

            if (idx >= size - 3)
                *ssize = -1;
            __syncthreads();

            if (*ssize == -1)
                return;

            idx += stride;
        }
    }
}