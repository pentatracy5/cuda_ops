#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernel.cuh>
#include <define.cuh>

namespace stream_schedule
{
    void depth_first(cudaStream_t* streams, const int num_streams, float* h_a, float* h_b, float* h_c, float* d_a, float* d_b, float* d_c, const int size)
    {
        const int size_per_stream = (size + num_streams - 1) / num_streams;
        const int shared_mem_bytes = 0;
        const int version = 0;
        for (int i = 0; i < num_streams; i++)
        {
            const int offset = i * size_per_stream;
            const int current_size = min(size_per_stream, size - offset);
            const int bytes_per_stream = current_size * sizeof(float);
            dim3 num_threads;
            dim3 threads_per_block;
            elementwise_add::get_kernel_launch_params(current_size, version, num_threads, threads_per_block);
            CUDA_CHECK(cudaMemcpyAsync(d_a + offset, h_a + offset, bytes_per_stream, cudaMemcpyHostToDevice, streams[i]));
            CUDA_CHECK(cudaMemcpyAsync(d_b + offset, h_b + offset, bytes_per_stream, cudaMemcpyHostToDevice, streams[i]));
            CUDA_LAUNCH_SHAREDMEM_STREAM(elementwise_add::kernels[version], num_threads, threads_per_block, shared_mem_bytes, streams[i])(d_a + offset, d_b + offset, d_c + offset, current_size);
            CUDA_KERNEL_LAUNCH_CHECK();
            CUDA_CHECK(cudaMemcpyAsync(h_c + offset, d_c + offset, bytes_per_stream, cudaMemcpyDeviceToHost, streams[i]));
        }
    }

    void breadth_first(cudaStream_t* streams, const int num_streams, float* h_a, float* h_b, float* h_c, float* d_a, float* d_b, float* d_c, const int size)
    {
        const int size_per_stream = (size + num_streams - 1) / num_streams;
        const int shared_mem_bytes = 0;
        const int version = 0;
        for (int i = 0; i < num_streams; i++)
        {
            const int offset = i * size_per_stream;
            const int current_size = min(size_per_stream, size - offset);
            const int bytes_per_stream = current_size * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(d_a + offset, h_a + offset, bytes_per_stream, cudaMemcpyHostToDevice, streams[i]));
            CUDA_CHECK(cudaMemcpyAsync(d_b + offset, h_b + offset, bytes_per_stream, cudaMemcpyHostToDevice, streams[i]));
        }
        for (int i = 0; i < num_streams; i++)
        {
            const int offset = i * size_per_stream;
            const int current_size = min(size_per_stream, size - offset);
            dim3 num_threads;
            dim3 threads_per_block;
            elementwise_add::get_kernel_launch_params(current_size, version, num_threads, threads_per_block);
            CUDA_LAUNCH_SHAREDMEM_STREAM(elementwise_add::kernels[version], num_threads, threads_per_block, shared_mem_bytes, streams[i])(d_a + offset, d_b + offset, d_c + offset, current_size);
            CUDA_KERNEL_LAUNCH_CHECK();
        }
        for (int i = 0; i < num_streams; i++)
        {
            const int offset = i * size_per_stream;
            const int current_size = min(size_per_stream, size - offset);
            const int bytes_per_stream = current_size * sizeof(float);
            CUDA_CHECK(cudaMemcpyAsync(h_c + offset, d_c + offset, bytes_per_stream, cudaMemcpyDeviceToHost, streams[i]));
        }
    }
}