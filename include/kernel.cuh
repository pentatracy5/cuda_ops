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

	__global__ void v3(float* input, float* output, const int size);

	__global__ void v4(float* input, float* output, const int size);

	__global__ void v5(float* input, float* output, const int size);

	__global__ void v6(float* input, float* output, const int size);

	__global__ void v7(float* input, float* output, const int size);

	__global__ void v8(float* input, float* output, const int size);

	using Kernel = decltype(&v0);

	static const Kernel kernels[]{ v0, v1, v2, v3, v4, v5, v6, v7, v8 };
}

namespace histogram
{
	int get_FLOPs(const int size);

	int get_bytes_transferred(const int size, const int bin_size);

	void get_kernel_launch_params(const int size, const int bin_size, const unsigned int version, int& num_threads, int& threads_per_block, int& shared_mem_bytes);

	__global__ void v0(float* data, int* bin, const int size, const int bin_size, const float lower_level, const float upper_level);

	__global__ void v1(float* data, int* bin, const int size, const int bin_size, const float lower_level, const float upper_level);

	using Kernel = decltype(&v0);

	static const Kernel kernels[]{ v0, v1 };
}

namespace copy_if
{
	int get_FLOPs(const int size);

	int get_bytes_transferred(const int size);

	void get_kernel_launch_params(const int size, const unsigned int version, int& num_threads, int& threads_per_block, int& shared_mem_bytes);

	__global__ void v0(float* src, float* dst, int* dst_size, const int size, const float compare);

	__global__ void v1(float* src, float* dst, int* dst_size, const int size, const float compare);

	__global__ void v2(float* src, float* dst, int* dst_size, const int size, const float compare);

	__global__ void v3(float* src, float* dst, int* dst_size, const int size, const float compare);

	using Kernel = decltype(&v0);

	static const Kernel kernels[]{ v0, v1, v2, v3 };
}

namespace elementwise_gelu
{
	int get_FLOPs(const int size);

	int get_bytes_transferred(const int size);

	void get_kernel_launch_params(const int size, const unsigned int version, int& num_threads, int& threads_per_block);

	__global__ void v0(float* input, float* output, const int size);

	using Kernel = decltype(&v0);

	static const Kernel kernels[]{ v0 };
}