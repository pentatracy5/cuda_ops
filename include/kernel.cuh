#pragma once

#include <cuda_runtime.h>

int get_FLOPs(const int size);

int get_bytes_transferred(const int size);

__global__ void add(float* a, float* b, float* c, const int size);

__global__ void vec2_add(float* a, float* b, float* c, const int size);

__global__ void vec4_add(float* a, float* b, float* c, const int size);

using AddKernel = decltype(&add);

static const AddKernel add_kernels[]{ add, vec2_add, vec4_add };