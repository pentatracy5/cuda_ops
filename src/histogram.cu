#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernel.cuh>
#include <define.cuh>

namespace histogram 
{
    int get_FLOPs(const int size)
    {
        return 2 * size;
    }

    int get_bytes_transferred(const int size, const int bin_size)
    {
        return size * sizeof(float) + 2 * bin_size * sizeof(int);
    }

    void get_kernel_launch_params(const int size, const int bin_size, const unsigned int version, int& num_threads, int& threads_per_block, int& shared_mem_bytes)
    {
        threads_per_block = 512;
        if (0 == version)
        {
            num_threads = (size / 4 + 31) / 32;
            shared_mem_bytes = 0;
        }
        else if (1 == version)
        {
            num_threads = (size / 4 + 31) / 32;
            shared_mem_bytes = bin_size * sizeof(int);
        }
        return;
    }

    __device__ __forceinline__ int compute_bin_id(const float val, const int bin_size, const float lower_level, const float upper_level)
    {
        return val / (upper_level - lower_level) * bin_size;
    }

    /**
     * @brief 直方图核函数版本0：直接使用全局内存原子操作。
     *
     * 每个线程一次处理4个浮点数（float4），通过原子加将每个值累加到对应的全局内存直方图 bin 中。
     * 所有线程直接竞争全局内存，在 bin 数量少或数据量大时原子冲突严重，性能可能受限。
     *
     * @param data        输入数据数组，包含 size 个 float 元素。
     * @param bin         输出直方图数组，长度为 bin_size，调用前需初始化为0。
     * @param size        输入数据中有效 float 元素的总数。
     * @param bin_size    直方图的 bin 数量，也决定共享内存局部直方图的大小。
     * @param lower_level 数值映射的下界，用于计算 bin 索引。
     * @param upper_level 数值映射的上界，用于计算 bin 索引。
     */
    __global__ void v0(float* data, int* bin, const int size, const int bin_size, const float lower_level, const float upper_level)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int stride = gridDim.x * blockDim.x * 4;
        while (idx < size - 3)
        {
            float4 reg = FETCH_FLOAT4(data[idx]);
            atomicAdd(&bin[compute_bin_id(reg.x, bin_size, lower_level, upper_level)], 1);
            atomicAdd(&bin[compute_bin_id(reg.y, bin_size, lower_level, upper_level)], 1);
            atomicAdd(&bin[compute_bin_id(reg.z, bin_size, lower_level, upper_level)], 1);
            atomicAdd(&bin[compute_bin_id(reg.w, bin_size, lower_level, upper_level)], 1);
            idx += stride;
        }
    }

    /**
     * @brief 直方图核函数版本1：使用共享内存局部直方图以减少全局原子操作冲突。
     *
     * 每个线程块在共享内存中维护一个私有的局部直方图，线程先将数据累加到共享内存，
     * 最后再将局部直方图原子累加到全局内存 bin 数组。这种方式显著减少了全局原子操作的次数，
     * 有效提升了高冲突场景下的性能，但需要额外共享内存。
     *
     * @note 启动核函数时必须通过第三个参数（动态共享内存大小）指定为 bin_size * sizeof(int) 字节。
     */
    __global__ void v1(float* data, int* bin, const int size, const int bin_size, const float lower_level, const float upper_level)
    {
        extern __shared__ int smem[];
        int tid = threadIdx.x;
        while (tid < bin_size)
        {
            smem[tid] = 0;
            tid += blockDim.x;
        }
        __syncthreads();

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx *= 4;
        int stride = gridDim.x * blockDim.x * 4;
        while (idx < size - 3)
        {
            float4 reg = FETCH_FLOAT4(data[idx]);
            atomicAdd(&smem[compute_bin_id(reg.x, bin_size, lower_level, upper_level)], 1);
            atomicAdd(&smem[compute_bin_id(reg.y, bin_size, lower_level, upper_level)], 1);
            atomicAdd(&smem[compute_bin_id(reg.z, bin_size, lower_level, upper_level)], 1);
            atomicAdd(&smem[compute_bin_id(reg.w, bin_size, lower_level, upper_level)], 1);
            idx += stride;
        }
        __syncthreads();

        tid = threadIdx.x;
        while (tid < bin_size)
        {
            atomicAdd(&bin[tid], smem[tid]);
            tid += blockDim.x;
        }
    }
}