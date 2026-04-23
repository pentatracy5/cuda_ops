#pragma once

#include <cuda_runtime.h>

namespace elementwise_add 
{
	int get_FLOPs(const int size);

	int get_bytes_transferred(const int size);

	int get_num_threads(const int size, const unsigned int version);

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

	int get_num_threads(const int size, const unsigned int version);

	int get_shared_mem_size(const int threads_per_block, const unsigned int version);

	__global__ void v0(float* input, float* output, const int size);

	__global__ void v1(float* input, float* output, const int size);

	using Kernel = decltype(&v0);

	static const Kernel kernels[]{ v0, v1 };
}