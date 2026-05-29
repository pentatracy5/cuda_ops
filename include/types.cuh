#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <CudaMirrorBuffer.cuh>
#include <Timer.cuh>

struct alignas(16) __half8
{
	__half val[8];
	__host__ __device__ inline const __half& operator[](int i) const { return val[i]; }
	__host__ __device__ inline __half& operator[](int i) { return val[i]; }
};

enum QuantizeType
{
	SYMMETRIC = 0,
	ASYMMETRIC = 1
};

enum RandType
{
	PSEUDO = 0,
	QUASI = 1
};

template <RandType rtype>
struct GetRandStateType
{
	using Type = curandState_t;
};

template <>
struct GetRandStateType<QUASI>
{
    using Type = curandStateScrambledSobol32_t;
};

template <>
struct GetRandStateType<PSEUDO>
{
    using Type = curandStatePhilox4_32_10_t;
};

template <typename T>
struct MaxOp
{
	__host__ __device__ T operator()(T a, T b) const { return max(a, b); }
};

template <typename T>
struct MinOp
{
	__host__ __device__ T operator()(T a, T b) const { return min(a, b); }
};

template <typename T>
struct AddOp
{
	__host__ __device__ T operator()(T a, T b) const { return a + b; }
};