#pragma once

#include <cuda_runtime.h>
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
class CudaMirrorBuffer
{
public:
    explicit CudaMirrorBuffer(size_t size): 
        size_(size), 
        h_ptr_(nullptr), 
        d_ptr_(nullptr) 
    {
        if (size_ == 0) 
            return;

        h_ptr_ = new T[size_];

        cudaMalloc((void**)&d_ptr_, size_ * sizeof(T));
        CHECK_CUDA_ERROR("cudaMalloc failed");

        CUDA_LAUNCH(construct_device_array<T>, (size_ + 255) / 256, 256)(d_ptr_, size_);
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR("construct device array failed");
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

    size_t size() const 
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
            cudaMemcpy(d_ptr_, h_ptr_, size_ * sizeof(T), cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR("syncHostToDevice failed");
        }
    }

    void to_host() const 
    {
        if (h_ptr_ && d_ptr_)
        {
            cudaMemcpy(h_ptr_, d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost);
            CHECK_CUDA_ERROR("syncDeviceToHost failed");
        }
    }

    void to_device_async(cudaStream_t stream = 0) const 
    {
        if (h_ptr_ && d_ptr_) 
        {
            cudaMemcpyAsync(d_ptr_, h_ptr_, size_ * sizeof(T), cudaMemcpyHostToDevice, stream);
            CHECK_CUDA_ERROR("syncHostToDeviceAsync failed");
        }
    }

    void to_host_async(cudaStream_t stream = 0) const 
    {
        if (h_ptr_ && d_ptr_)
        {
            cudaMemcpyAsync(h_ptr_, d_ptr_, size_ * sizeof(T), cudaMemcpyDeviceToHost, stream);
            CHECK_CUDA_ERROR("syncDeviceToHostAsync failed");
        }
    }

    void resize(size_t newSize) 
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

private:
    void release() 
    {
        delete [] h_ptr_;
        h_ptr_ = nullptr;
        if (d_ptr_) 
        {
            cudaFree(d_ptr_);
            CHECK_CUDA_ERROR("cudaFree failed");
            d_ptr_ = nullptr;
        }
        size_ = 0;
    }

private:
    size_t size_;
    T* h_ptr_;
    T* d_ptr_;
};