#pragma once

#include <cuda_runtime.h>
#include <types.cuh>
#include <config.cuh>

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

	__global__ void v0(__half* input, __half* output, const int size);

	__global__ void v1(__half* input, __half* output, const int size);

	__global__ void v2(__half* input, __half* output, const int size);

	__global__ void v_ref(__half* input, __half* output, const int size);

	using Kernel = decltype(&v0);

	static const Kernel kernels[]{ v0, v1, v2, v_ref };
}

namespace stream_schedule
{
	void depth_first(cudaStream_t* streams, const int num_streams, float* h_a, float* h_b, float* h_c, float* d_a, float* d_b, float* d_c, const int size);

	void breadth_first(cudaStream_t* streams, const int num_streams, float* h_a, float* h_b, float* h_c, float* d_a, float* d_b, float* d_c, const int size);

	using Kernel = decltype(&depth_first);

	static const Kernel kernels[]{ depth_first, breadth_first };
}

namespace quantize
{
	int get_FLOPs(const int rows, const int cols);

	int get_bytes_transferred(const int rows, const int cols);

	void get_kernel_launch_params(const int rows, const int cols, const unsigned int version, int& num_threads, int& threads_per_block, int& shared_mem_bytes);

	template <QuantizeType qtype>
	__global__ void v0(float* d_input, int8_t* d_output, const int rows, const int cols, const float qmin, const float qmax);

	using Kernel = decltype(&v0<QUANTIZETYPE>);

	static const Kernel kernels[]{ v0<QUANTIZETYPE> };
}

namespace softmax
{
	int get_FLOPs(const int rows, const int cols);

	int get_bytes_transferred(const int rows, const int cols);

	void get_kernel_launch_params(const int rows, const int cols, const unsigned int version, int& num_threads, int& threads_per_block, int& shared_mem_bytes);

	__global__ void v0(float* d_input, float* d_output, const int rows, const int cols);

	using Kernel = decltype(&v0);

	static const Kernel kernels[]{ v0 };
}

namespace gemv_col_major
{
	int get_FLOPs(const int rows, const int cols);

	int get_bytes_transferred(const int rows, const int cols);

	void get_kernel_launch_params(const int rows, const int cols, const unsigned int version, int& num_threads, int& threads_per_block, int& shared_mem_bytes);

	__global__ void v0(float* m, float* v, float* d_output, const int rows, const int cols);

	__global__ void v1(float* m, float* v, float* d_output, const int rows, const int cols);

	using Kernel = decltype(&v0);

	static const Kernel kernels[]{ v0, v1 };
}