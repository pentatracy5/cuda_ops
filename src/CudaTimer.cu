#include <CudaTimer.cuh>
#include <define.cuh>

CudaTimer::CudaTimer()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
}

CudaTimer::~CudaTimer()
{
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void CudaTimer::tic(cudaStream_t stream/* = 0*/)
{
    cudaEventRecord(start, stream);
}

float CudaTimer::toc(cudaStream_t stream/* = 0*/)
{
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}