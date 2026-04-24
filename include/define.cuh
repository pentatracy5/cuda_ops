#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#ifndef __INTELLISENSE__

#define NUM_GRIDS(n_threads, threads_per_block)                                                         ((n_threads + threads_per_block - 1) / threads_per_block)
#define CUDA_LAUNCH(kernel, n_threads, threads_per_block)										        kernel<<<NUM_GRIDS(n_threads, threads_per_block), threads_per_block>>>
#define CUDA_LAUNCH_SHAREDMEM(kernel, n_threads, threads_per_block, shared_mem_bytes)				    kernel<<<NUM_GRIDS(n_threads, threads_per_block), threads_per_block, shared_mem_bytes>>>
#define CUDA_LAUNCH_SHAREDMEM_STREAM(kernel, n_threads, threads_per_block, shared_mem_bytes, stream)	kernel<<<NUM_GRIDS(n_threads, threads_per_block), threads_per_block, shared_mem_bytes, stream>>>

#else

#define CUDA_LAUNCH(kernel, n_threads, threads_per_block)										        kernel
#define CUDA_LAUNCH_SHAREDMEM(kernel, n_threads, threads_per_block, shared_mem_bytes)			    	kernel
#define CUDA_LAUNCH_SHAREDMEM_STREAM(kernel, n_threads, threads_per_block, shared_mem_bytes, stream)	kernel
float atomicAdd(float* address, float val);
void __syncthreads();

#endif

#define CHECK_CUDA_ERROR(msg) check_cuda_error_impl((msg), __FILE__, __LINE__)
inline void check_cuda_error_impl(const char* msg, const char* file, int line)
{
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err) + " at file " + std::string(file) + " line " + std::to_string(line));
    }
}

#define FETCH_FLOAT2(var) (reinterpret_cast<float2*>(&(var))[0])
#define FETCH_FLOAT4(var) (reinterpret_cast<float4*>(&(var))[0])