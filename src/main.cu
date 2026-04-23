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
	CudaMirrorBuffer<float> ref(N);

	random_init_array(a.host(), N);
	random_init_array(b.host(), N);
	a.to_device();
	b.to_device();

	int num_threads = elementwise_add::get_num_threads(N, version);

	for (size_t i = 0; i < WARMUP; i++)
	{
		CUDA_LAUNCH(elementwise_add::kernels[version], num_threads, THREADS_PER_BLOCK)(a.device(), b.device(), c.device(), N);
		CHECK_CUDA_ERROR("run kernel failed");
	}

	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	for (size_t i = 0; i < NREPEATS; i++)
	{
		CUDA_LAUNCH(elementwise_add::kernels[version], num_threads, THREADS_PER_BLOCK)(a.device(), b.device(), c.device(), N);
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
	cout << "Memory Bandwidth: " << elementwise_add::get_bytes_transferred(N) / 1e6 / elapsed_time << " GB/s" << endl;
	cout << "Achieved GFLOPS: " << elementwise_add::get_FLOPs(N) / 1e6 / elapsed_time << " GFLOPS" << endl;
}

void run_reduce_sum(unsigned int version)
{
	CudaMirrorBuffer<float> input(N);
	CudaMirrorBuffer<float> output(1);
	CudaMirrorBuffer<float> ref(1);

	random_init_array(input.host(), N);
	constant_val_init_array(output.host(), 1, 0);
	constant_val_init_array(ref.host(), 1, 0);
	input.to_device();
	output.to_device();

	int num_threads = reduce_sum::get_num_threads(N, version);

	for (size_t i = 0; i < WARMUP; i++)
	{
		CUDA_LAUNCH(reduce_sum::kernels[version], num_threads, THREADS_PER_BLOCK)(input.device(), output.device(), N);
		CHECK_CUDA_ERROR("run kernel failed");
	}

	float milliseconds = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	for (size_t i = 0; i < NREPEATS; i++)
	{
		CUDA_LAUNCH(reduce_sum::kernels[version], num_threads, THREADS_PER_BLOCK)(input.device(), output.device(), N);
		CHECK_CUDA_ERROR("run kernel failed");
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);

	output.to_host();

	float* input_host = input.host();
	float* ref_host = ref.host();
	for (size_t i = 0; i < WARMUP + NREPEATS; i++)
		for (int idx = 0; idx < N; idx++)
			ref_host[0] += input_host[idx];

	compare_array(output.host(), ref_host, 1, TOLERANCE);

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