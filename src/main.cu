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

void run_elementwise_add(unsigned int version)
{
	CudaMirrorBuffer<float> a(N);
	CudaMirrorBuffer<float> b(N);
	CudaMirrorBuffer<float> c(N);

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

	float* a_host = a.host();
	float* b_host = b.host();
	for (int i = 0; i < N; i++)
		a_host[i] += b_host[i];

	compare_array(c.host(), a_host, N, TOLERANCE);

	float elapsed_time = milliseconds / NREPEATS;
	cout << "Memory Bandwidth: " << elementwise_add::get_bytes_transferred(N) / 1e6 / elapsed_time << " GB/s" << endl;
	cout << "Achieved GFLOPS: " << elementwise_add::get_FLOPs(N) / 1e6 / elapsed_time << " GFLOPS" << endl;
}

void run_reduce_sum(unsigned int version)
{
	CudaMirrorBuffer<float> input(N);
	CudaMirrorBuffer<float> output(1);

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

	//避免大数吞小数
	float* input_host = input.host();
	int remainder = N & 1;
	int stride = N >> 1;
	do
	{
		for (size_t i = 0; i < stride; i++)
			input_host[i] += input_host[i + stride];
		if (remainder)
			input_host[0] += input_host[2 * stride];
		remainder = stride & 1;
		stride = stride >> 1;
	} while (stride > 0);

	compare_array(output.host(), input_host, 1, TOLERANCE);

	float elapsed_time = milliseconds / NREPEATS;
	cout << "Memory Bandwidth: " << reduce_sum::get_bytes_transferred(N) / 1e6 / elapsed_time << " GB/s" << endl;
	cout << "Achieved GFLOPS: " << reduce_sum::get_FLOPs(N) / 1e6 / elapsed_time << " GFLOPS" << endl;
}

void run(unsigned int type, unsigned int version)
{
	if (0 == type)
	{
		cout << "elementwise add version " << version << endl;
		run_elementwise_add(version);
		cout << endl;
	}
	else if (1 == type)
	{
		cout << "reduce sum version " << version << endl;
		run_reduce_sum(version);
		cout << endl;
	}
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

	run(type, version);

	return 0;
}