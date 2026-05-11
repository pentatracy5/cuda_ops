#include <test.cuh>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <CudaMirrorBuffer.cuh>
#include <kernel.cuh>
#include <define.cuh>
#include <config.cuh>
#include <utils.cuh>
#include <cub/cub.cuh>

using std::cout;
using std::endl;
using std::sort;

namespace elementwise_add
{
	void run(unsigned int version)
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

	void test(unsigned int version)
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
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

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

		compare_array(c.host(), ref.host(), N, TOLERANCETIGHT);

		cout << "elementwise add\t\tversion " << version << "\tREF" << endl;
		cout << "Memory Bandwidth:\t" << elementwise_add::get_bytes_transferred(N) / 1e6 / time << " GB/s\t" << elementwise_add::get_bytes_transferred(N) / 1e9 / time_ref << " GB/s" << endl;
		cout << "Achieved GFLOPS:\t" << elementwise_add::get_FLOPs(N) / 1e6 / time << " GFLOPS\t" << elementwise_add::get_FLOPs(N) / 1e9 / time_ref << " GFLOPS" << endl;
		cout << endl;
	}
}

namespace reduce_sum
{
	void run(unsigned int version)
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

			int temp_size = NUM_GRIDS(num_threads, threads_per_block);
			CudaMirrorBuffer<float> temp_storage(temp_size);

			if (8 == version)
			{
				for (size_t i = 0; i < NREPEATS; i++)
				{
					CUDA_LAUNCH_SHAREDMEM(reduce_sum::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), temp_storage.device(), N);
					CHECK_CUDA_ERROR("run kernel failed");
					CUDA_LAUNCH_SHAREDMEM(reduce_sum::kernels[version], threads_per_block, threads_per_block, shared_mem_bytes)(temp_storage.device(), output.device(), temp_size);
					CHECK_CUDA_ERROR("run kernel failed");
				}
			}
			else
			{
				for (size_t i = 0; i < NREPEATS; i++)
				{
					output.memset(0);
					CUDA_LAUNCH_SHAREDMEM(reduce_sum::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), output.device(), N);
					CHECK_CUDA_ERROR("run kernel failed");
				}
			}
		}
	}

	void test(unsigned int version)
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

		int temp_size = NUM_GRIDS(num_threads, threads_per_block);
		CudaMirrorBuffer<float> temp_storage(temp_size);

		if (8 == version)
		{
			for (size_t i = 0; i < WARMUP; i++)
			{
				CUDA_LAUNCH_SHAREDMEM(reduce_sum::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), temp_storage.device(), N);
				CHECK_CUDA_ERROR("run kernel failed");
				CUDA_LAUNCH_SHAREDMEM(reduce_sum::kernels[version], threads_per_block, threads_per_block, shared_mem_bytes)(temp_storage.device(), output.device(), temp_size);
				CHECK_CUDA_ERROR("run kernel failed");
			}
		}
		else
		{
			for (size_t i = 0; i < WARMUP; i++)
			{
				output.memset(0);
				CUDA_LAUNCH_SHAREDMEM(reduce_sum::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), output.device(), N);
				CHECK_CUDA_ERROR("run kernel failed");
			}
		}

		float milliseconds = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		if (8 == version)
		{
			for (size_t i = 0; i < NREPEATS; i++)
			{
				CUDA_LAUNCH_SHAREDMEM(reduce_sum::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), temp_storage.device(), N);
				CHECK_CUDA_ERROR("run kernel failed");
				CUDA_LAUNCH_SHAREDMEM(reduce_sum::kernels[version], threads_per_block, threads_per_block, shared_mem_bytes)(temp_storage.device(), output.device(), temp_size);
				CHECK_CUDA_ERROR("run kernel failed");
			}
		}
		else
		{
			for (size_t i = 0; i < NREPEATS; i++)
			{
				output.memset(0);
				CUDA_LAUNCH_SHAREDMEM(reduce_sum::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), output.device(), N);
				CHECK_CUDA_ERROR("run kernel failed");
			}
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

		cudaEventRecord(start);

		for (size_t i = 0; i < NREPEATS; i++)
		{
			cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, input.device(), ref.device(), N);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaFree(d_temp_storage);
		CHECK_CUDA_ERROR("cudaFree failed");
		ref.to_host();
		float time_ref = milliseconds / NREPEATS;

		compare_array(output.host(), ref.host(), 1, TOLERANCELOOSE);

		cout << "reduce sum\t\tversion " << version << "\tREF" << endl;
		cout << "Memory Bandwidth:\t" << reduce_sum::get_bytes_transferred(N) / 1e6 / time << " GB/s\t" << reduce_sum::get_bytes_transferred(N) / 1e6 / time_ref << " GB/s" << endl;
		cout << "Achieved GFLOPS:\t" << reduce_sum::get_FLOPs(N) / 1e6 / time << " GFLOPS\t" << reduce_sum::get_FLOPs(N) / 1e6 / time_ref << " GFLOPS" << endl;
		cout << endl;
	}
}

namespace histogram
{
	void run(unsigned int version)
	{
		CudaMirrorBuffer<float> data(N);
		CudaMirrorBuffer<int> bin(BINSIZE);

		random_init_array(data.host(), N);
		data.to_device();

		if constexpr (PROFILEREF)
		{
			void* d_temp_storage = nullptr;
			size_t temp_storage_bytes = 0;
			cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, data.device(), bin.device(), BINSIZE + 1, LOWERLEVEL, UPPERLEVEL, N);
			CHECK_CUDA_ERROR("run kernel failed");
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			CHECK_CUDA_ERROR("cudaMalloc failed");

			for (size_t i = 0; i < NREPEATS; i++)
			{
				cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, data.device(), bin.device(), BINSIZE + 1, LOWERLEVEL, UPPERLEVEL, N);
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
			histogram::get_kernel_launch_params(N, BINSIZE, version, num_threads, threads_per_block, shared_mem_bytes);

			for (size_t i = 0; i < NREPEATS; i++)
			{
				bin.memset(0);
				CUDA_LAUNCH_SHAREDMEM(histogram::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(data.device(), bin.device(), N, BINSIZE, LOWERLEVEL, UPPERLEVEL);
				CHECK_CUDA_ERROR("run kernel failed");
			}
		}
	}

	void test(unsigned int version)
	{
		CudaMirrorBuffer<float> data(N);
		CudaMirrorBuffer<int> bin(BINSIZE);
		CudaMirrorBuffer<int> ref(BINSIZE);

		random_init_array(data.host(), N);
		data.to_device();

		int num_threads;
		int threads_per_block;
		int shared_mem_bytes;
		histogram::get_kernel_launch_params(N, BINSIZE, version, num_threads, threads_per_block, shared_mem_bytes);

		for (size_t i = 0; i < WARMUP; i++)
		{
			bin.memset(0);
			CUDA_LAUNCH_SHAREDMEM(histogram::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(data.device(), bin.device(), N, BINSIZE, LOWERLEVEL, UPPERLEVEL);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		float milliseconds = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		for (size_t i = 0; i < NREPEATS; i++)
		{
			bin.memset(0);
			CUDA_LAUNCH_SHAREDMEM(histogram::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(data.device(), bin.device(), N, BINSIZE, LOWERLEVEL, UPPERLEVEL);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);

		bin.to_host();
		float time = milliseconds / NREPEATS;

		void* d_temp_storage = nullptr;
		size_t temp_storage_bytes = 0;
		cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, data.device(), ref.device(), BINSIZE + 1, LOWERLEVEL, UPPERLEVEL, N);
		CHECK_CUDA_ERROR("run kernel failed");
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		CHECK_CUDA_ERROR("cudaMalloc failed");

		for (size_t i = 0; i < WARMUP; i++)
		{
			cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, data.device(), ref.device(), BINSIZE + 1, LOWERLEVEL, UPPERLEVEL, N);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaEventRecord(start);

		for (size_t i = 0; i < NREPEATS; i++)
		{
			cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes, data.device(), ref.device(), BINSIZE + 1, LOWERLEVEL, UPPERLEVEL, N);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaFree(d_temp_storage);
		CHECK_CUDA_ERROR("cudaFree failed");
		ref.to_host();
		float time_ref = milliseconds / NREPEATS;

		compare_array(bin.host(), ref.host(), BINSIZE, 0);

		cout << "histogram\t\tversion " << version << "\tREF" << endl;
		cout << "Memory Bandwidth:\t" << histogram::get_bytes_transferred(N, BINSIZE) / 1e6 / time << " GB/s\t" << histogram::get_bytes_transferred(N, BINSIZE) / 1e6 / time_ref << " GB/s" << endl;
		cout << "Achieved GFLOPS:\t" << histogram::get_FLOPs(N) / 1e6 / time << " GFLOPS\t" << histogram::get_FLOPs(N) / 1e6 / time_ref << " GFLOPS" << endl;
		cout << endl;
	}
}

namespace copy_if
{
	struct LessThan
	{
		float compare;
		__host__ __device__ __forceinline__ LessThan(float compare) : compare(compare) {}
		__host__ __device__ __forceinline__ bool operator()(const float& a) const { return (a < compare); }
	};

	void run(unsigned int version)
	{
		CudaMirrorBuffer<float> src(N);
		CudaMirrorBuffer<float> dst(N);
		CudaMirrorBuffer<int> dst_size(1);

		random_init_array(src.host(), N);
		src.to_device();

		if constexpr (PROFILEREF)
		{
			LessThan select_op(COMPARE);
			void* d_temp_storage = nullptr;
			size_t temp_storage_bytes = 0;
			cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, src.device(), dst.device(), dst_size.device(), N, select_op);
			CHECK_CUDA_ERROR("run kernel failed");
			cudaMalloc(&d_temp_storage, temp_storage_bytes);
			CHECK_CUDA_ERROR("cudaMalloc failed");

			for (size_t i = 0; i < NREPEATS; i++)
			{
				cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, src.device(), dst.device(), dst_size.device(), N, select_op);
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
			copy_if::get_kernel_launch_params(N, version, num_threads, threads_per_block, shared_mem_bytes);

			for (size_t i = 0; i < NREPEATS; i++)
			{
				dst_size.memset(0);
				CUDA_LAUNCH_SHAREDMEM(copy_if::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(src.device(), dst.device(), dst_size.device(), N, COMPARE);
				CHECK_CUDA_ERROR("run kernel failed");
			}
		}
	}

	void test(unsigned int version)
	{
		CudaMirrorBuffer<float> src(N);
		CudaMirrorBuffer<float> dst(N);
		CudaMirrorBuffer<int> dst_size(1);
		CudaMirrorBuffer<float> ref(N);
		CudaMirrorBuffer<int> ref_size(1);

		random_init_array(src.host(), N);
		src.to_device();

		int num_threads;
		int threads_per_block;
		int shared_mem_bytes;
		copy_if::get_kernel_launch_params(N, version, num_threads, threads_per_block, shared_mem_bytes);

		for (size_t i = 0; i < WARMUP; i++)
		{
			dst_size.memset(0);
			CUDA_LAUNCH_SHAREDMEM(copy_if::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(src.device(), dst.device(), dst_size.device(), N, COMPARE);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		float milliseconds = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		for (size_t i = 0; i < NREPEATS; i++)
		{
			dst_size.memset(0);
			CUDA_LAUNCH_SHAREDMEM(copy_if::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(src.device(), dst.device(), dst_size.device(), N, COMPARE);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);

		dst.to_host();
		dst_size.to_host();
		float time = milliseconds / NREPEATS;

		LessThan select_op(COMPARE);
		void* d_temp_storage = nullptr;
		size_t temp_storage_bytes = 0;
		cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, src.device(), ref.device(), ref_size.device(), N, select_op);
		CHECK_CUDA_ERROR("run kernel failed");
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		CHECK_CUDA_ERROR("cudaMalloc failed");

		for (size_t i = 0; i < WARMUP; i++)
		{
			cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, src.device(), ref.device(), ref_size.device(), N, select_op);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaEventRecord(start);

		for (size_t i = 0; i < NREPEATS; i++)
		{
			cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, src.device(), ref.device(), ref_size.device(), N, select_op);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaFree(d_temp_storage);
		CHECK_CUDA_ERROR("cudaFree failed");
		ref.to_host();
		ref_size.to_host();
		float time_ref = milliseconds / NREPEATS;

		compare_array(dst_size.host(), ref_size.host(), 1, 0);
		sort(dst.host(), dst.host() + dst_size.host()[0]);
		sort(ref.host(), ref.host() + ref_size.host()[0]);
		compare_array(dst.host(), ref.host(), dst_size.host()[0], 0.f);

		cout << "copy if\t\t\tversion " << version << "\tREF" << endl;
		cout << "Memory Bandwidth:\t" << copy_if::get_bytes_transferred(N) / 1e6 / time << " GB/s\t" << copy_if::get_bytes_transferred(N) / 1e6 / time_ref << " GB/s" << endl;
		cout << "Achieved GFLOPS:\t" << copy_if::get_FLOPs(N) / 1e6 / time << " GFLOPS\t" << copy_if::get_FLOPs(N) / 1e6 / time_ref << " GFLOPS" << endl;
		cout << endl;
	}
}

namespace elementwise_gelu
{
	void run(unsigned int version) {}

	void test(unsigned int version)
	{
		CudaMirrorBuffer<__half> input(N);
		CudaMirrorBuffer<__half> output(N);
		CudaMirrorBuffer<__half> ref(N);

		random_init_array(input.host(), N);
		input.to_device();

		int num_threads;
		int threads_per_block;
		elementwise_gelu::get_kernel_launch_params(N, version, num_threads, threads_per_block);

		for (size_t i = 0; i < WARMUP; i++)
		{
			CUDA_LAUNCH(elementwise_gelu::kernels[version], num_threads, threads_per_block)(input.device(), output.device(), N);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		float milliseconds = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		for (size_t i = 0; i < NREPEATS; i++)
		{
			CUDA_LAUNCH(elementwise_gelu::kernels[version], num_threads, threads_per_block)(input.device(), output.device(), N);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);

		output.to_host();
		float time = milliseconds / NREPEATS;

		int ref_version = sizeof(elementwise_gelu::kernels) / sizeof(elementwise_gelu::kernels[0]) - 1;
		elementwise_gelu::get_kernel_launch_params(N, ref_version, num_threads, threads_per_block);

		for (size_t i = 0; i < WARMUP; i++)
		{
			CUDA_LAUNCH(elementwise_gelu::kernels[ref_version], num_threads, threads_per_block)(input.device(), ref.device(), N);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaEventRecord(start);

		for (size_t i = 0; i < NREPEATS; i++)
		{
			CUDA_LAUNCH(elementwise_gelu::kernels[ref_version], num_threads, threads_per_block)(input.device(), ref.device(), N);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		ref.to_host();
		float time_ref = milliseconds / NREPEATS;

		compare_array(output.host(), ref.host(), N, TOLERANCETIGHT);

		cout << "elementwise gelu\tversion " << version << "\t\tREF" << endl;
		cout << "Memory Bandwidth:\t" << elementwise_gelu::get_bytes_transferred(N) / 1e6 / time << " GB/s\t\t" << elementwise_gelu::get_bytes_transferred(N) / 1e6 / time_ref << " GB/s" << endl;
		cout << "Achieved GFLOPS:\t" << elementwise_gelu::get_FLOPs(N) / 1e6 / time << " GFLOPS(FP16)\t" << elementwise_gelu::get_FLOPs(N) / 1e6 / time_ref << " GFLOPS(FP16)" << endl;
		cout << endl;
	}
}