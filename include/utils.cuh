#pragma once

#include <cuda_runtime.h>
#include <random>
#include <iostream>
#include <config.cuh>

template<typename T>
void random_init_array(T* array, int size)
{
	std::mt19937 engine(std::random_device{}());
	std::uniform_real_distribution<float> dist(LOWERLEVEL, UPPERLEVEL);
	for (int i = 0; i < size; i++)
		array[i] = (T)dist(engine);
}

template<typename T>
void compare_array(T* output, T* ref, const int size, const float tolerance)
{
	for (int i = 0; i < size; i++)
		if (fabs(output[i] - ref[i]) > tolerance)
		{
			std::cout << "Error: output(" << i << ") = " << float(output[i]) << ", but expected " << float(ref[i]) << std::endl;
			return;
		}
	return;
}

template <typename T>
__device__ __forceinline__ T clamp(T val, T lo, T hi)
{
	return min(max(val, lo), hi);
}

template <int warp_size, typename T, template <typename> class Op>
__device__ __forceinline__ T shuffle_warp_reduce(T x, Op<T> op)
{
	constexpr unsigned int mask = (1ULL << warp_size) - 1;
	if (32 <= warp_size)    x = op(x, __shfl_down_sync(mask, x, 16));
	if (16 <= warp_size)    x = op(x, __shfl_down_sync(mask, x, 8));
	if (8 <= warp_size)     x = op(x, __shfl_down_sync(mask, x, 4));
	if (4 <= warp_size)     x = op(x, __shfl_down_sync(mask, x, 2));
	if (2 <= warp_size)     x = op(x, __shfl_down_sync(mask, x, 1));
	return x;
}