#pragma once

#include <random>
#include <iostream>
#include <config.cuh>
#include <cuda_fp16.h>

using std::mt19937;
using std::random_device;
using std::uniform_real_distribution;
using std::cout;
using std::endl;

template<typename T>
void random_init_array(T* array, int size)
{
	mt19937 engine(random_device{}());
	uniform_real_distribution<float> dist(LOWERLEVEL, UPPERLEVEL);
	for (int i = 0; i < size; i++)
		array[i] = (T)dist(engine);
}

template<typename T>
void compare_array(T* output, T* ref, const int size, const float tolerance)
{
	for (int i = 0; i < size; i++)
		if (fabs(output[i] - ref[i]) > tolerance)
		{
			cout << "Error: output(" << i << ") = " << float(output[i]) << ", but expected " << float(ref[i]) << endl;
			return;
		}
	return;
}

struct alignas(16) __half8
{
	__half val[8];
	__host__ __device__ inline const __half& operator[](int i) const { return val[i]; }
	__host__ __device__ inline __half& operator[](int i) { return val[i]; }
};
