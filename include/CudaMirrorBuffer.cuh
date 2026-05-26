#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <malloc.h>
#include <utility>
#include <cassert>
#include <define.cuh>

template<typename T>
__global__ void construct_device_array(T* ptr, size_t n) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) 
        new(ptr + idx) T();
}

template<typename T>
__global__ void constant_val_set_kernel(T* ptr, size_t n, T val)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        ptr[idx] = val;
}

template<typename T>
class CudaMirrorBuffer
{
public:
    explicit CudaMirrorBuffer(unsigned int size, cudaStream_t stream = 0):
        size_(size), 
        h_ptr_(nullptr), 
        d_ptr_(nullptr) 
    {
        if (size_ == 0) 
            return;

        CUDA_CHECK(cudaHostAlloc((void**)&h_ptr_, size_ * sizeof(T), cudaHostAllocDefault));
        for (size_t i = 0; i < size; i++)
            new(h_ptr_ + i) T();

        CUDA_CHECK(cudaMalloc((void**)&d_ptr_, size_ * sizeof(T)));

        CUDA_LAUNCH_SHAREDMEM_STREAM(construct_device_array<T>, dim3{ size_ }, dim3{ 512 }, 0, stream)(d_ptr_, size_);
        CUDA_KERNEL_LAUNCH_CHECK();
    }

    ~CudaMirrorBuffer() 
    {
        release();
    }

    CudaMirrorBuffer(const CudaMirrorBuffer&) = delete;

    CudaMirrorBuffer& operator=(const CudaMirrorBuffer&) = delete;

    CudaMirrorBuffer(CudaMirrorBuffer&& other) noexcept: 
        size_(other.size_), 
        h_ptr_(other.h_ptr_), 
        d_ptr_(other.d_ptr_) 
    {
        other.size_ = 0;
        other.h_ptr_ = nullptr;
        other.d_ptr_ = nullptr;
    }

    CudaMirrorBuffer& operator=(CudaMirrorBuffer&& other) noexcept 
    {
        if (this != &other)
        {
            release();
            size_ = other.size_;
            h_ptr_ = other.h_ptr_;
            d_ptr_ = other.d_ptr_;
            other.size_ = 0;
            other.h_ptr_ = nullptr;
            other.d_ptr_ = nullptr;
        }
        return *this;
    }

    T* host() 
    {
        assert(!empty() && "Accessing host pointer of empty buffer");
        return h_ptr_;
    }

    const T* host() const 
    {
        assert(!empty() && "Accessing host pointer of empty buffer");
        return h_ptr_;
    }

    T* device() 
    {
        assert(!empty() && "Accessing device pointer of empty buffer");
        return d_ptr_;
    }

    const T* device() const 
    {
        assert(!empty() && "Accessing device pointer of empty buffer");
        return d_ptr_;
    }

    unsigned int size() const 
    { 
        return size_;
    }

    bool empty() const noexcept
    {
        return size_ == 0;
    }

    void to_device() const 
    {
        if (h_ptr_ && d_ptr_) 
        {
            CUDA_CHECK(cudaMemcpy(d_ptr_, h_ptr_, size_ * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    void to_host() const 
    {
        if (h_ptr_ && d_ptr_)
        {
            CUDA_CHECK(cudaMemcpy(h_ptr_, d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
        }
    }

    void to_device_async(cudaStream_t stream = 0) const 
    {
        if (h_ptr_ && d_ptr_) 
        {
            CUDA_CHECK(cudaMemcpyAsync(d_ptr_, h_ptr_, size_ * sizeof(T), cudaMemcpyHostToDevice, stream));
        }
    }

    void to_host_async(cudaStream_t stream = 0) const 
    {
        if (h_ptr_ && d_ptr_)
        {
            CUDA_CHECK(cudaMemcpyAsync(h_ptr_, d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
        }
    }

    void resize(unsigned int newSize) 
    {
        if (newSize == size_) 
            return;

        CudaMirrorBuffer<T> temp(newSize);
        swap(temp);
    }

    void swap(CudaMirrorBuffer& other) noexcept 
    {
        std::swap(size_, other.size_);
        std::swap(h_ptr_, other.h_ptr_);
        std::swap(d_ptr_, other.d_ptr_);
    }

    void memset(int value, cudaStream_t stream = 0)
    {
        if (empty())
            return;
        std::memset(h_ptr_, value, size_ * sizeof(T));
        CUDA_CHECK(cudaMemsetAsync(d_ptr_, value, size_ * sizeof(T), stream));
    }

    void constant_val_set(const T& val, cudaStream_t stream = 0)
    {
        if (empty())
            return;
        std::fill(h_ptr_, h_ptr_ + size_, val);
        CUDA_LAUNCH_SHAREDMEM_STREAM(constant_val_set_kernel<T>, dim3{ size_ }, dim3{ 512 }, 0, stream)(d_ptr_, size_, val);
        CUDA_KERNEL_LAUNCH_CHECK();
    }

private:
    void release() 
    {
        if (h_ptr_)
        {
            for (size_t i = 0; i < size_; ++i)
                h_ptr_[i].~T();
            CUDA_CHECK(cudaFreeHost(h_ptr_));
            h_ptr_ = nullptr;
        }
        if (d_ptr_) 
        {
            CUDA_CHECK(cudaFree(d_ptr_));
            d_ptr_ = nullptr;
        }
        size_ = 0;
    }

private:
    unsigned int size_;
    T* h_ptr_;
    T* d_ptr_;
};