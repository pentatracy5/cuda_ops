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
        else if (7 <= version && 9 > version)
        {
            num_threads = (size / 4 + 31) / 32;
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
    template<int warp_size>
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
    //template<int warp_size>
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
    //template<int warp_size>
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

    /**
     * @brief 基于向量化加载和共享内存树形归约的并行求和内核（使用 warp 同步优化块内最后一级归约）
     *
     * @note 相比 v3 的改进：
     *       - 在块内共享内存归约的最后阶段（剩余元素为 64 时），采用 warp 级别的同步归约（warp_reduce），
     *         利用 `__syncwarp` 进行精确同步，替代了代价更高的 `__syncthreads`，减少了块内同步开销，
     *         提升了单 warp 内的归约效率。
     *       - 同时保留向量化加载（float4），维持高带宽利用率。
     *
     * @note 存在的开销与缺陷：
     *       - 当 block 数量很多（输入规模超大）时，仍需多次 atomicAdd 累加各 block 的部分和，
     *         原子操作冲突与串行化的开销依然存在。
     *       - 全局累加阶段仍可能出现“大数吞小数”现象，影响浮点求和精度。
     */
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

    /**
     * @brief 基于向量化加载和两级 warp 归约的并行求和内核
     *
     * @note 相比 v4 的改进：
     *       - 引入两级 warp 归约：先以 32 个线程为一组进行 warp 级归约，将结果存入共享内存；
     *         通过 `__syncthreads` 同步后，再用更小的 mini-warp（4 线程）对共享内存中的部分和
     *         进行二次归约。这种结构进一步减少了全局 atomicAdd 之前所需的同步和共享内存访问次数，
     *         降低了块内归约的尾部延迟。
     *
     * @note 存在的开销与缺陷：
     *       - 多 block 场景下 atomicAdd 的竞争与全局精度问题仍然存在。
     */
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

    /**
     * @brief 在单个 warp 内使用 warp shuffle 操作对浮点值进行归约求和。
     *
     * @tparam warp_size  参与归约的线程数，必须为 2 的幂且 ≤ 32。
     * 
     * @param x  当前线程持有的待归约浮点值。
     * 
     * @return   对于 laneid == 0 的线程，返回归约后的总和。
     *
     * @note 使用 `__shfl_down_sync` 指令在 warp 内的线程间直接交换寄存器值并累加，
     *       完全消除对共享内存的依赖，从而获得更低的延迟和更高的带宽效率。
     *       该函数要求从 lane 0 开始共 `warp_size` 个线程同时调用，且这些线程必须属于
     *       同一个 warp。内部通过 `__shfl_down_sync(mask, ...)` 进行线程间数据交换，
     *       该指令对 mask 中的线程具有隐式同步效果，因此无需额外的 `__syncwarp`。
     */
    template<int warp_size>
    __device__ __forceinline__ float shuffle_warp_reduce_sum(float x)
    {
        constexpr unsigned int mask = (1ULL << warp_size) - 1;
        if (32 <= warp_size)    x += __shfl_down_sync(mask, x, 16);
        if (16 <= warp_size)    x += __shfl_down_sync(mask, x, 8);
        if (8 <= warp_size)     x += __shfl_down_sync(mask, x, 4);
        if (4 <= warp_size)     x += __shfl_down_sync(mask, x, 2);
        if (2 <= warp_size)     x += __shfl_down_sync(mask, x, 1);
        return x;
    }

    /**
     * @brief 基于向量化加载和 warp shuffle 操作的并行求和内核
     *
     * @note 相比 v5 的改进：
     *       - 使用 `__shfl_down_sync` 进行 warp 内归约，数据直接在寄存器间交换，
     *         完全消除了 warp 内归约对共享内存的依赖，显著减少了共享内存访问延迟和带宽占用。
     *       - 仅在 warp 间通信时才使用共享内存，最后的 block 内归约也通过 shuffle 完成，
     *         最大程度发挥了寄存器通信的高带宽、低延迟优势。
     *
     * @note 存在的开销与缺陷：
     *       - 多个 block 的部分和最终仍需通过 atomicAdd 累加，冲突开销和全局精度问题依然存在。
     */
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

        constexpr int warp_size = 32;
        x = shuffle_warp_reduce_sum<warp_size>(x);
        if (0 == (tid & (warp_size - 1)))
            smem[tid >> 5] = x;
        __syncthreads();

        constexpr int mini_warp_size = 16;
        if (tid < mini_warp_size)
            x = shuffle_warp_reduce_sum<mini_warp_size>(smem[tid]);

        if (0 == tid)
            atomicAdd(output, x);
    }

    /**
     * @brief 基于 grid-stride loop 和 warp shuffle 的并行求和内核
     *
     * @note 相比 v6 的改进：
     *       - 引入 grid-stride loop：每个线程按 `gridDim.x * blockDim.x * 4` 的步长遍历全局内存，
     *         可累加多个 float4 块，从而支持使用远少于数据元素数量的线程完成归约。
     *         这大幅减少了网格启动规模和线程总数，降低了调度开销，同时提高了单线程的计算密度
     *         和指令级并行度，一定程度上也减少了后续 atomicAdd 的次数，并且缓解累加精度问题。
     *       - 继续沿用高效的 warp shuffle 归约，保持了优秀的块内和 warp 间通信性能。
     *
     * @note 存在的开销与缺陷：
     *       - 最终多个 block 的部分和仍需通过 atomicAdd 进行全局归约，在超大输入规模下可能成为性能瓶颈，
     *         并且同样面临浮点累加精度（大数吞小数）的问题。
     */
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

        constexpr int warp_size = 32;
        x = shuffle_warp_reduce_sum<warp_size>(x);
        if (0 == (tid & (warp_size - 1)))
            smem[tid >> 5] = x;
        __syncthreads();

        constexpr int mini_warp_size = 16;
        if (tid < mini_warp_size)
            x = shuffle_warp_reduce_sum<mini_warp_size>(smem[tid]);

        if (0 == tid)
            atomicAdd(output, x);
    }

    /**
     * @brief 基于 grid-stride loop 和 warp shuffle 的并行求和内核（消除全局原子操作）
     *
     * @note 相比 v7 的改进：
     *       - 每个 block 不再通过 atomicAdd 将部分和累加到全局输出，而是直接将部分和
     *         写入 output[blockIdx.x]，完全消除了多 block 竞争全局内存地址带来的原子操作
     *         开销和串行化问题，在多 block 场景下具备更好的并行可扩展性。
     *
     * @note 存在的开销与缺陷：
     *       - 输出变为按 block 独立存放的一组部分和（共 gridDim.x 个元素），调用者必须
     *         在 kernel 执行完成后通过额外步骤（如在 CPU 端循环累加，或启动另一个归约 kernel）
     *         合并所有部分和才能获得最终结果，增加了整体算法复杂度。
     *       - output 数组需要至少 gridDim.x 个 float 元素的空间，且求和结果的精度取决于
     *         后续合并的方式，若直接简单累加仍可能出现“大数吞小数”现象。
     */
    __global__ void v8(float* input, float* output, const int size)
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

        constexpr int warp_size = 32;
        x = shuffle_warp_reduce_sum<warp_size>(x);
        if (0 == (tid & (warp_size - 1)))
            smem[tid >> 5] = x;
        __syncthreads();

        constexpr int mini_warp_size = 16;
        if (tid < mini_warp_size)
            x = shuffle_warp_reduce_sum<mini_warp_size>(smem[tid]);

        if (0 == tid)
            output[blockIdx.x] = x;
    }
}