#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <types.cuh>

#define NBS_PER_DIM(n_threads, threads_per_block)   ((n_threads + threads_per_block - 1) / threads_per_block)
#define N_BLOCKS(n_threads, threads_per_block)      (dim3{NBS_PER_DIM(n_threads.x, threads_per_block.x), NBS_PER_DIM(n_threads.y, threads_per_block.y), NBS_PER_DIM(n_threads.z, threads_per_block.z)})

#ifndef __INTELLISENSE__

#define CUDA_LAUNCH(kernel, n_threads, threads_per_block)										        kernel<<<N_BLOCKS(n_threads, threads_per_block), threads_per_block>>>
#define CUDA_LAUNCH_SHAREDMEM(kernel, n_threads, threads_per_block, shared_mem_bytes)				    kernel<<<N_BLOCKS(n_threads, threads_per_block), threads_per_block, shared_mem_bytes>>>
#define CUDA_LAUNCH_SHAREDMEM_STREAM(kernel, n_threads, threads_per_block, shared_mem_bytes, stream)	kernel<<<N_BLOCKS(n_threads, threads_per_block), threads_per_block, shared_mem_bytes, stream>>>

#else

#define CUDA_LAUNCH(kernel, n_threads, threads_per_block)										        kernel
#define CUDA_LAUNCH_SHAREDMEM(kernel, n_threads, threads_per_block, shared_mem_bytes)			    	kernel
#define CUDA_LAUNCH_SHAREDMEM_STREAM(kernel, n_threads, threads_per_block, shared_mem_bytes, stream)	kernel
float atomicAdd(float* address, float val);
int atomicAdd(int* address, int val);
void __syncthreads();
void __syncwarp(unsigned mask = 0xffffffff);
template <typename T>
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width = 32);
unsigned __activemask();
int __ffs(int x);
int __popc(unsigned int x);
template <typename T>
T __shfl_sync(unsigned mask, T var, int srcLane, int width = 32);

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
#define FETCH_HALF2(var) (reinterpret_cast<__half2*>(&(var))[0])
#define FETCH_HALF8(var) (reinterpret_cast<__half8*>(&(var))[0])
#define FETCH_CHAR4(var) (reinterpret_cast<char4*>(&(var))[0])