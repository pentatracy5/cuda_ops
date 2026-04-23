#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <kernel.cuh>
#include <define.cuh>

int get_FLOPs(const int size)
{
    return size;
}

int get_bytes_transferred(const int size)
{
    return 3 * size * sizeof(float);
}

__global__ void add(float* a, float* b, float* c, const int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)   return;
    c[idx] = a[idx] + b[idx];
}

__global__ void vec2_add(float* a, float* b, float* c, const int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx *= 2;
    if (idx >= size - 1)   return;
    float2 reg_a = FETCH_FLOAT2(a[idx]);
    float2 reg_b = FETCH_FLOAT2(b[idx]);
    float2 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    FETCH_FLOAT2(c[idx]) = reg_c;
}

__global__ void vec4_add(float* a, float* b, float* c, const int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    idx *= 4;
    if (idx >= size - 3)   return;
    float4 reg_a = FETCH_FLOAT4(a[idx]);
    float4 reg_b = FETCH_FLOAT4(b[idx]);
    float4 reg_c;
    reg_c.x = reg_a.x + reg_b.x;
    reg_c.y = reg_a.y + reg_b.y;
    reg_c.z = reg_a.z + reg_b.z;
    reg_c.w = reg_a.w + reg_b.w;
    FETCH_FLOAT4(c[idx]) = reg_c;
}