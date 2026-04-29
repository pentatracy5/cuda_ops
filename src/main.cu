#include <iostream>
#include <sstream>
#include <chrono>
#include <CudaMirrorBuffer.cuh>
#include <kernel.cuh>
#include <define.cuh>
#include <config.cuh>
#include <utils.cuh>
#include <cub/cub.cuh>

using std::cout;
using std::endl;
using std::istringstream;
using std::cerr;

void run_elementwise_add(unsigned int version)
{
	CudaMirrorBuffer<float> a(N);
	CudaMirrorBuffer<float> b(N);
	CudaMirrorBuffer<float> c(N);

	random_init_array(a.host(), N);
	random_init_array(b.host(), N);
	a.to_device();
	b.to_device();

	if constexpr (PROFILEREF)
	{
		float* a_host = a.host();
		float* b_host = b.host();
		float* c_host = c.host();

		for (int i = 0; i < NREPEATS; i++)
			for (int j = 0; j < N; j++)
				c_host[j] = a_host[j] + b_host[j];
	}
	else
	{
		int num_threads;
		int threads_per_block;
		elementwise_add::get_kernel_launch_params(N, version, num_threads, threads_per_block);

		for (size_t i = 0; i < NREPEATS; i++)
		{
			CUDA_LAUNCH(elementwise_add::kernels[version], num_threads, threads_per_block)(a.device(), b.device(), c.device(), N);
			CHECK_CUDA_ERROR("run kernel failed");
		}
	}
}

void test_elementwise_add(unsigned int version)
{
	CudaMirrorBuffer<float> a(N);
	CudaMirrorBuffer<float> b(N);
	CudaMirrorBuffer<float> c(N);
	CudaMirrorBuffer<float> ref(N);

	random_init_array(a.host(), N);
	random_init_array(b.host(), N);
	a.to_device();
	b.to_device();

	int num_threads;
	int threads_per_block;
	elementwise_add::get_kernel_launch_params(N, version, num_threads, threads_per_block);

	for (size_t i = 0; i < WARMUP; i++)
	{
		CUDA_LAUNCH(elementwise_add::kernels[version], num_threads, threads_per_block)(a.device(), b.device(), c.device(), N);
		CHECK_CUDA_ERROR("run kernel failed");
	}

	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	for (size_t i = 0; i < NREPEATS; i++)
	{
		CUDA_LAUNCH(elementwise_add::kernels[version], num_threads, threads_per_block)(a.device(), b.device(), c.device(), N);
		CHECK_CUDA_ERROR("run kernel failed");
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	c.to_host();
	float time = milliseconds / NREPEATS;

	float* a_host = a.host();
	float* b_host = b.host();
	float* ref_host = ref.host();

	for (int i = 0; i < WARMUP; i++)
		for (int j = 0; j < N; j++)
			ref_host[j] = a_host[j] + b_host[j];

	auto begin = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < NREPEATS; i++)
		for (int j = 0; j < N; j++)
			ref_host[j] = a_host[j] + b_host[j];

	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - begin;

	double time_ref = elapsed.count() / NREPEATS;

	compare_array(c.host(), ref.host(), N, TOLERANCE);

	cout << "elementwise add\t\tversion " << version << "\tREF" << endl;
	cout << "Memory Bandwidth:\t" << elementwise_add::get_bytes_transferred(N) / 1e6 / time << " GB/s\t" << elementwise_add::get_bytes_transferred(N) / 1e9 / time_ref << " GB/s" << endl;
	cout << "Achieved GFLOPS:\t" << elementwise_add::get_FLOPs(N) / 1e6 / time << " GFLOPS\t" << elementwise_add::get_FLOPs(N) / 1e9 / time_ref << " GFLOPS" << endl;
	cout << endl;
}

void run_reduce_sum(unsigned int version)
{
	CudaMirrorBuffer<float> input(N);
	CudaMirrorBuffer<float> output(1);

	random_init_array(input.host(), N);
	input.to_device();

	if constexpr (PROFILEREF)
	{
		void* d_temp_storage = nullptr;
		size_t temp_storage_bytes = 0;
		cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input.device(), output.device(), N);
		CHECK_CUDA_ERROR("run kernel failed");
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		CHECK_CUDA_ERROR("cudaMalloc failed");

		for (size_t i = 0; i < NREPEATS; i++)
		{
			cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input.device(), output.device(), N);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaFree(d_temp_storage);
		CHECK_CUDA_ERROR("cudaFree failed");
	}
	else
	{
		int num_threads;
		int threads_per_block;
		int shared_mem_bytes;
		reduce_sum::get_kernel_launch_params(N, version, num_threads, threads_per_block, shared_mem_bytes);

		for (size_t i = 0; i < NREPEATS; i++)
		{
			output.memset(0);
			CUDA_LAUNCH_SHAREDMEM(reduce_sum::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), output.device(), N);
			CHECK_CUDA_ERROR("run kernel failed");
		}
	}
}

void test_reduce_sum(unsigned int version)
{
	CudaMirrorBuffer<float> input(N);
	CudaMirrorBuffer<float> output(1);
	CudaMirrorBuffer<float> ref(1);

	random_init_array(input.host(), N);
	input.to_device();

	int num_threads;
	int threads_per_block;
	int shared_mem_bytes;
	reduce_sum::get_kernel_launch_params(N, version, num_threads, threads_per_block, shared_mem_bytes);

	for (size_t i = 0; i < WARMUP; i++)
	{
		output.memset(0);
		CUDA_LAUNCH_SHAREDMEM(reduce_sum::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), output.device(), N);
		CHECK_CUDA_ERROR("run kernel failed");
	}

	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	for (size_t i = 0; i < NREPEATS; i++)
	{
		output.memset(0);
		CUDA_LAUNCH_SHAREDMEM(reduce_sum::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), output.device(), N);
		CHECK_CUDA_ERROR("run kernel failed");
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	output.to_host();
	float time = milliseconds / NREPEATS;

	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input.device(), ref.device(), N);
	CHECK_CUDA_ERROR("run kernel failed");
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	CHECK_CUDA_ERROR("cudaMalloc failed");

	for (size_t i = 0; i < WARMUP; i++)
	{
		cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input.device(), ref.device(), N);
		CHECK_CUDA_ERROR("run kernel failed");
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	for (size_t i = 0; i < NREPEATS; i++)
	{
		cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input.device(), ref.device(), N);
		CHECK_CUDA_ERROR("run kernel failed");
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaFree(d_temp_storage);
	CHECK_CUDA_ERROR("cudaFree failed");
	ref.to_host();
	float time_ref = milliseconds / NREPEATS;

	compare_array(output.host(), ref.host(), 1, TOLERANCE);

	cout << "reduce sum\t\tversion " << version << "\tREF" << endl;
	cout << "Memory Bandwidth:\t" << reduce_sum::get_bytes_transferred(N) / 1e6 / time << " GB/s\t" << reduce_sum::get_bytes_transferred(N) / 1e6 / time_ref << " GB/s" << endl;
	cout << "Achieved GFLOPS:\t" << reduce_sum::get_FLOPs(N) / 1e6 / time << " GFLOPS\t" << reduce_sum::get_FLOPs(N) / 1e6 / time_ref << " GFLOPS" << endl;
	cout << endl;
}

void run(unsigned int type, unsigned int version)
{
	if (0 == type)
		run_elementwise_add(version);
	else if (1 == type)
		run_reduce_sum(version);
}

void test(unsigned int type, unsigned int version)
{
	if (0 == type)
		test_elementwise_add(version);
	else if (1 == type)
		test_reduce_sum(version);
}

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		cout << "Error: require 2 arguments, but " << argc - 1 << " provided." << endl;
		return 1;
	}

	unsigned int type;
	istringstream iss1(argv[1]);
	if (!(iss1 >> type)) {
		cerr << "Error: invalid ops type." << endl;
		return 1;
	}

	unsigned int version;
	istringstream iss2(argv[2]);
	if (!(iss2 >> version)) {
		cerr << "Error: invalid ops version." << endl;
		return 1;
	}

	if constexpr (PROFILE)
		run(type, version);
	else
		test(type, version);

	return 0;
}