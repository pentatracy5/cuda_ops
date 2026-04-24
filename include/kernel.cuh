#pragma once

#include <cuda_runtime.h>

namespace elementwise_add 
{
	int get_FLOPs(const int size);

	int get_bytes_transferred(const int size);

	void get_kernel_launch_params(const int size, const unsigned int version, int& num_threads, int& threads_per_block);

	__global__ void no_vectorize(float* a, float* b, float* c, const int size);

	__global__ void vectorize2(float* a, float* b, float* c, const int size);

	__global__ void vectorize4(float* a, float* b, float* c, const int size);

	using Kernel = decltype(&no_vectorize);

	static const Kernel kernels[]{ no_vectorize, vectorize2, vectorize4 };
}

namespace reduce_sum
{
	int get_FLOPs(const int size);

	int get_bytes_transferred(const int size);

	void get_kernel_launch_params(const int size, const unsigned int version, int& num_threads, int& threads_per_block, int& shared_mem_bytes);

	__global__ void v0(float* input, float* output, const int size);

	__global__ void v1(float* input, float* output, const int size);

	__global__ void v2(float* input, float* output, const int size);

	using Kernel = decltype(&v0);

	static const Kernel kernels[]{ v0, v1, v2 };
}