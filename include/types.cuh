#pragma once

#include <cuda_fp16.h>
#include <CudaMirrorBuffer.cuh>

struct alignas(16) __half8
{
	__half val[8];
	__host__ __device__ inline const __half& operator[](int i) const { return val[i]; }
	__host__ __device__ inline __half& operator[](int i) { return val[i]; }
};

enum QuantizeType
{
	SYMMETRIC_PER_TENSOR = 0,
	SYMMETRIC_PER_CHANNEL = 1,
	ASYMMETRIC_PER_TENSOR = 2,
	ASYMMETRIC_PER_CHANNEL = 3
};