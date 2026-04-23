#include <iostream>
#include <sstream>
#include <CudaMirrorBuffer.cuh>
#include <kernel.cuh>
#include <define.cuh>
#include <config.cuh>
#include <utils.cuh>

using std::cout;
using std::endl;
using std::istringstream;
using std::cerr;

void Run(unsigned int version)
{
	CudaMirrorBuffer<float> a(N);
	CudaMirrorBuffer<float> b(N);
	CudaMirrorBuffer<float> c(N);
	CudaMirrorBuffer<float> ref(N);

	random_init_array(a.host(), N);
	random_init_array(b.host(), N);
	a.to_device();
	b.to_device();

	int num_threads = 0;
	if (0 == version)
		num_threads = N;
	else if (1 == version)
		num_threads = N / 2;
	else if (2 == version)
		num_threads = N / 4;

	for (size_t i = 0; i < WARMUP; i++)
	{
		CUDA_LAUNCH(add_kernels[version], num_threads, THREADS_PER_BLOCK)(a.device(), b.device(), c.device(), N);
		CHECK_CUDA_ERROR("run kernel failed");
	}

	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	for (size_t i = 0; i < NREPEATS; i++)
	{
		CUDA_LAUNCH(add_kernels[version], num_threads, THREADS_PER_BLOCK)(a.device(), b.device(), c.device(), N);
		CHECK_CUDA_ERROR("run kernel failed");
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	c.to_host();

	float* a_host = a.host();
	float* b_host = b.host();
	float* ref_host = ref.host();
	for (int i = 0; i < N; i++)
		ref_host[i] = a_host[i] + b_host[i];

	compare_array(c.host(), ref_host, N, TOLERANCE);

	float elapsed_time = milliseconds / NREPEATS;
	cout << "Memory Bandwidth: " << get_bytes_transferred(N) / 1e6 / elapsed_time << " GB/s" << endl;
	cout << "Achieved GFLOPS: " << get_FLOPs(N) / 1e6 / elapsed_time << " GFLOPS" << endl;
}

int main(int argc, char* argv[])
{
	//unsigned int version;
	//if (argc != 2)
	//{
	//	cout << "Error: require 1 arguments, but " << argc - 1 << " provided." << endl;
	//	return 1;
	//}

	//istringstream iss1(argv[1]);
	//if (!(iss1 >> version)) {
	//	cerr << "Error: invalid ops version." << endl;
	//	return 1;
	//}

	//Run(version);

	int total_versions = sizeof(add_kernels) / sizeof(add_kernels[0]);
	for (size_t i = 0; i < total_versions; i++)
	{
		cout << "kernel version " << i << endl;
		Run(i);
		cout << endl;
	}

	return 0;
}