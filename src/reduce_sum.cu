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
        else if (3 <= version && 6 > version)
        {
            num_threads = size / 4;
            shared_mem_bytes = threads_per_block * sizeof(float);
        }
        else if (6 == version)
        {
            num_threads = size / 4;
            shared_mem_bytes = threads_per_block / 32 * sizeof(float);
        }
        else if (7 == version)
        {
            num_threads = size / 128;
            shared_mem_bytes = threads_per_block / 32 * sizeof(float);
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
        extern __shared__ float smem[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int tid = threadIdx.x;
        smem[tid] = idx >= size ? 0.f : input[idx];
        __syncthreads();
        int stride = blockDim.x >> 1;
        while (stride > 0)
        {
            if (tid < stride)
                smem[tid] += smem[tid + stride];
            __syncthreads();
            stride = stride >> 1;
        }
        if (0 == tid)
            atomicAdd(output, smem[0]);
    }

    // 错误的写法，因为CUDA要求一个 block 内的所有线程必须同时到达同一个 __syncthreads()，如果有一部分线程提前 return ，可能会导致死锁或未定义行为。
    // 见 https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#synchronization-functions
    // "__syncthreads() is allowed in conditional code but only if the conditional evaluates identically across the entire thread block, 
    // otherwise the code execution is likely to hang or produce unintended side effects."
    //__global__ void v2(float* input, float* output, const int size)
    //{
    //    extern __shared__ float smem[];
    //    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //    int tid = threadIdx.x;
    //    smem[tid] = idx >= size ? 0.f : input[idx];
    //    __syncthreads();
    //    int stride = blockDim.x >> 1;
    //    while (stride > 0)
    //    {
    //        if (tid >= stride)
    //            return;
    //        input[idx] += input[idx + stride];
    //        __syncthreads();
    //        stride = stride >> 1;
    //    }
    //    atomicAdd(output, input[idx]);
    //}

    // 既然 shared memory 和 L1 cache 物理上是同一块内存，那么能否不使用 shared memory ，直接靠 GPU 的缓存机制，实现 reduce 加速呢？
    // 下面给出不使用 shared memory 版本的算法
    // 
    //__global__ void v2(float* input, float* output, const int size)
    //{
    //    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    //    int tid = threadIdx.x;
    //    __syncthreads();
    //    int stride = blockDim.x >> 1;
    //    while (stride > 0)
    //    {
    //        if (tid < stride)
    //            input[idx] += input[idx + stride];
    //        __syncthreads();
    //        stride = stride >> 1;
    //    }
    //    if (0 == tid)
    //        atomicAdd(output, input[idx]);
    //}
    // 
    // 理论上来说，不使用 shared memory 的做法，大概率也能保证累加的过程发生在 L1 cache 中，但是因为 L1 cache 其实是 global memory 的镜像，
    // 所以在 L1 cache 上的所有写入，最终都会写回到 global memory 中，这就带来两个问题
    //  - 这种写回是不必要的，会增加算法的带宽，从而拖慢算法
    //  - 写回的操作会改变原始输入的 input 数据，影响后续数据的复用

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
        extern __shared__ float smem[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int tid = threadIdx.x;
        smem[tid] = 0.f;
        if (idx < size - 3)
        {
            float4 reg = FETCH_FLOAT4(input[idx]);
            smem[tid] = reg.x + reg.y + reg.z + reg.w;
        }
        __syncthreads();
        int stride = blockDim.x >> 1;
        while (stride > 0)
        {
            if (tid < stride)
                smem[tid] += smem[tid + stride];
            __syncthreads();
            stride = stride >> 1;
        }
        if (0 == tid)
            atomicAdd(output, smem[0]);
    }

    /**
     * @brief 在单个 warp 内对共享内存中的浮点数据进行归约（求和）。
     *
     * @tparam warp_size  warp 中参与归约的线程数，必须是 2 的幂且 ≤ 32（CUDA 最大 warp 大小）。
     *
     * @note 该函数适用于 Volta 及之后的 GPU 架构（SM 7.0+）。在这些架构上，同一 warp 内的线程
     *       不再保证自动执行隐式同步，因此必须显式使用 `__syncwarp()` 来保证内存操作的顺序与可见性。
     *       函数在每次 `load` 和 `store` 之后都调用了 `__syncwarp(mask)`，确保：
     *         - 所有参与线程在读取其他线程的写入前完成存储；
     *         - 所有参与线程看到一致的内存状态。
     *       其中 `mask` 覆盖所有参与归约的线程（即 `(1ULL << warp_size) - 1`）。
     *        `__syncwarp(mask)` 要求所有mask中指定（即对应bit为1）的线程都参与该同步，否则行为未定义。
     *
     * @see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
     */
    template <int warp_size>
    __device__ __forceinline__ void warp_reduce(float* smem, int tid)
    {
        constexpr unsigned int mask = (1ULL << warp_size) - 1;
        float x;
        if (32 <= warp_size)    { x = smem[tid + 32];   __syncwarp(mask);   smem[tid] += x; __syncwarp(mask); }
        if (16 <= warp_size)    { x = smem[tid + 16];   __syncwarp(mask);   smem[tid] += x; __syncwarp(mask); }
        if (8 <= warp_size)     { x = smem[tid + 8];    __syncwarp(mask);   smem[tid] += x; __syncwarp(mask); }
        if (4 <= warp_size)     { x = smem[tid + 4];    __syncwarp(mask);   smem[tid] += x; __syncwarp(mask); }
        if (2 <= warp_size)     { x = smem[tid + 2];    __syncwarp(mask);   smem[tid] += x; __syncwarp(mask); }
        if (1 <= warp_size)     { x = smem[tid + 1];    __syncwarp(mask);   smem[tid] += x; }
    }

    // 错误的写法，会被编译器优化，导致计算结果不对
    //template <int warp_size>
    //__device__ __forceinline__ void warp_reduce(float* smem, int tid)
    //{
    //    if (32 <= warp_size)
    //        smem[tid] += smem[tid + 32];
    //    if (16 <= warp_size)
    //        smem[tid] += smem[tid + 16];
    //    if (8 <= warp_size)
    //        smem[tid] += smem[tid + 8];
    //    if (4 <= warp_size)
    //        smem[tid] += smem[tid + 4];
    //    if (2 <= warp_size)
    //        smem[tid] += smem[tid + 2];
    //    if (1 <= warp_size)
    //        smem[tid] += smem[tid + 1];
    //}

    // 不推荐的写法，因为不同版本的编译器不能保证volatile关键字的行为一致
    //template <int warp_size>
    //__device__ __forceinline__ void warp_reduce(volatile float* smem, int tid)
    //{
    //    if (32 <= warp_size)
    //        smem[tid] += smem[tid + 32];
    //    if (16 <= warp_size)
    //        smem[tid] += smem[tid + 16];
    //    if (8 <= warp_size)
    //        smem[tid] += smem[tid + 8];
    //    if (4 <= warp_size)
    //        smem[tid] += smem[tid + 4];
    //    if (2 <= warp_size)
    //        smem[tid] += smem[tid + 2];
    //    if (1 <= warp_size)
    //        smem[tid] += smem[tid + 1];
    //}

    __global__ void v4(float* input, float* output, const int size)
    {
        extern __shared__ float smem[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int tid = threadIdx.x;
        smem[tid] = 0.f;
        if (idx < size - 3)
        {
            float4 reg = FETCH_FLOAT4(input[idx]);
            smem[tid] = reg.x + reg.y + reg.z + reg.w;
        }
        __syncthreads();
        constexpr int warp_size = 32;
        int stride = blockDim.x >> 1;
        while (stride > warp_size)
        {
            if (tid < stride)
                smem[tid] += smem[tid + stride];
            __syncthreads();
            stride = stride >> 1;
        }
        if (tid < warp_size)
            warp_reduce<warp_size>(smem, tid);
        if (0 == tid)
            atomicAdd(output, smem[0]);
    }

    __global__ void v5(float* input, float* output, const int size)
    {
        extern __shared__ float smem[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int tid = threadIdx.x;
        smem[tid] = 0.f;
        if (idx < size - 3)
        {
            float4 reg = FETCH_FLOAT4(input[idx]);
            smem[tid] = reg.x + reg.y + reg.z + reg.w;
        }
        __syncthreads();

        constexpr int warp_size = 32;
        constexpr int warp_reduce_range = warp_size << 1;
        int warp_reduce_range_id = tid & (warp_reduce_range - 1);
        if (warp_reduce_range_id < warp_size)
            warp_reduce<warp_size>(smem, tid);
        __syncthreads();

        constexpr int mini_warp_size = 4;
        constexpr int mini_warp_reduce_range = mini_warp_size << 1;
        if (tid < mini_warp_reduce_range)
        {
            constexpr unsigned int mask = (1ULL << mini_warp_reduce_range) - 1;
            smem[tid] = smem[tid << 6];
            __syncwarp(mask);
        }
        if (tid < mini_warp_size)
            warp_reduce<mini_warp_size>(smem, tid);

        if (0 == tid)
            atomicAdd(output, smem[0]);
    }

    template <int warp_size>
    __device__ __forceinline__ float shuffle_warp_reduce(float x)
    {
        constexpr unsigned int mask = (1ULL << (warp_size << 1)) - 1;
        if (16 <= warp_size)    x += __shfl_down_sync(mask, x, 16);
        if (8 <= warp_size)     x += __shfl_down_sync(mask, x, 8);
        if (4 <= warp_size)     x += __shfl_down_sync(mask, x, 4);
        if (2 <= warp_size)     x += __shfl_down_sync(mask, x, 2);
        if (1 <= warp_size)     x += __shfl_down_sync(mask, x, 1);
        return x;
    }

    __global__ void v6(float* input, float* output, const int size)
    {
        extern __shared__ float smem[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int tid = threadIdx.x;
        float x = 0.f;
        if (idx < size - 3)
        {
            float4 reg = FETCH_FLOAT4(input[idx]);
            x = reg.x + reg.y + reg.z + reg.w;
        }

        constexpr int warp_size = 16;
        constexpr int warp_reduce_range = warp_size << 1;
        x = shuffle_warp_reduce<warp_size>(x);
        if (0 == (tid & (warp_reduce_range - 1)))
            smem[tid >> 5] = x;
        __syncthreads();

        constexpr int mini_warp_size = 8;
        constexpr int mini_warp_reduce_range = mini_warp_size << 1;
        if (tid < mini_warp_reduce_range)
            x = shuffle_warp_reduce<mini_warp_size>(smem[tid]);

        if (0 == tid)
            atomicAdd(output, x);
    }

    __global__ void v7(float* input, float* output, const int size)
    {
        extern __shared__ float smem[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int tid = threadIdx.x;
        float x = 0.f;
        int stride = gridDim.x * blockDim.x * 4;
        while (idx < size - 3)
        {
            float4 reg = FETCH_FLOAT4(input[idx]);
            x += reg.x + reg.y + reg.z + reg.w;
            idx += stride;
        }

        constexpr int warp_size = 16;
        constexpr int warp_reduce_range = warp_size << 1;
        x = shuffle_warp_reduce<warp_size>(x);
        if (0 == (tid & (warp_reduce_range - 1)))
            smem[tid >> 5] = x;
        __syncthreads();

        constexpr int mini_warp_size = 8;
        constexpr int mini_warp_reduce_range = mini_warp_size << 1;
        if (tid < mini_warp_reduce_range)
            x = shuffle_warp_reduce<mini_warp_size>(smem[tid]);

        if (0 == tid)
            atomicAdd(output, x);
    }
}