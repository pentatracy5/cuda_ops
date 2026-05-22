#include <test.cuh>
#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <types.cuh>
#include <kernel.cuh>
#include <define.cuh>
#include <config.cuh>
#include <utils.cuh>
#include <cub/cub.cuh>

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
			dim3 num_threads;
			dim3 threads_per_block;
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

		dim3 num_threads;
		dim3 threads_per_block;
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

		double time_ref = elapsed.count() / NREPEATS * 1e3;

		compare_array(c.host(), ref.host(), N, 0);

		std::cout << "elementwise add\t\tversion " << version << "\tREF" << std::endl;
		std::cout << "Memory Bandwidth:\t" << elementwise_add::get_bytes_transferred(N) / 1e6 / time << " GB/s\t" << elementwise_add::get_bytes_transferred(N) / 1e6 / time_ref << " GB/s" << std::endl;
		std::cout << "Achieved GFLOPS:\t" << elementwise_add::get_FLOPs(N) / 1e6 / time << " GFLOPS\t" << elementwise_add::get_FLOPs(N) / 1e6 / time_ref << " GFLOPS" << std::endl;
		std::cout << std::endl;
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
			dim3 num_threads;
			dim3 threads_per_block;
			int shared_mem_bytes;
			reduce_sum::get_kernel_launch_params(N, version, num_threads, threads_per_block, shared_mem_bytes);

			int temp_size = N_BLOCKS(num_threads, threads_per_block).x;
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

		dim3 num_threads;
		dim3 threads_per_block;
		int shared_mem_bytes;
		reduce_sum::get_kernel_launch_params(N, version, num_threads, threads_per_block, shared_mem_bytes);

		int temp_size = N_BLOCKS(num_threads, threads_per_block).x;
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

		std::cout << "reduce sum\t\tversion " << version << "\tREF" << std::endl;
		std::cout << "Memory Bandwidth:\t" << reduce_sum::get_bytes_transferred(N) / 1e6 / time << " GB/s\t" << reduce_sum::get_bytes_transferred(N) / 1e6 / time_ref << " GB/s" << std::endl;
		std::cout << "Achieved GFLOPS:\t" << reduce_sum::get_FLOPs(N) / 1e6 / time << " GFLOPS\t" << reduce_sum::get_FLOPs(N) / 1e6 / time_ref << " GFLOPS" << std::endl;
		std::cout << std::endl;
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
			dim3 num_threads;
			dim3 threads_per_block;
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

		dim3 num_threads;
		dim3 threads_per_block;
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

		std::cout << "histogram\t\tversion " << version << "\tREF" << std::endl;
		std::cout << "Memory Bandwidth:\t" << histogram::get_bytes_transferred(N, BINSIZE) / 1e6 / time << " GB/s\t" << histogram::get_bytes_transferred(N, BINSIZE) / 1e6 / time_ref << " GB/s" << std::endl;
		std::cout << "Achieved GFLOPS:\t" << histogram::get_FLOPs(N) / 1e6 / time << " GFLOPS\t" << histogram::get_FLOPs(N) / 1e6 / time_ref << " GFLOPS" << std::endl;
		std::cout << std::endl;
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
			dim3 num_threads;
			dim3 threads_per_block;
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

		dim3 num_threads;
		dim3 threads_per_block;
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
		std::sort(dst.host(), dst.host() + dst_size.host()[0]);
		std::sort(ref.host(), ref.host() + ref_size.host()[0]);
		compare_array(dst.host(), ref.host(), dst_size.host()[0], 0.f);

		std::cout << "copy if\t\t\tversion " << version << "\tREF" << std::endl;
		std::cout << "Memory Bandwidth:\t" << copy_if::get_bytes_transferred(N) / 1e6 / time << " GB/s\t" << copy_if::get_bytes_transferred(N) / 1e6 / time_ref << " GB/s" << std::endl;
		std::cout << "Achieved GFLOPS:\t" << copy_if::get_FLOPs(N) / 1e6 / time << " GFLOPS\t" << copy_if::get_FLOPs(N) / 1e6 / time_ref << " GFLOPS" << std::endl;
		std::cout << std::endl;
	}
}

namespace elementwise_gelu
{
	void run(unsigned int version)
	{
		CudaMirrorBuffer<__half> input(N);
		CudaMirrorBuffer<__half> output(N);

		random_init_array(input.host(), N);
		input.to_device();

		dim3 num_threads;
		dim3 threads_per_block;
		if constexpr (PROFILEREF)
			version = sizeof(elementwise_gelu::kernels) / sizeof(elementwise_gelu::kernels[0]) - 1;
		elementwise_gelu::get_kernel_launch_params(N, version, num_threads, threads_per_block);

		for (size_t i = 0; i < NREPEATS; i++)
		{
			CUDA_LAUNCH(elementwise_gelu::kernels[version], num_threads, threads_per_block)(input.device(), output.device(), N);
			CHECK_CUDA_ERROR("run kernel failed");
		}
	}

	void test(unsigned int version)
	{
		CudaMirrorBuffer<__half> input(N);
		CudaMirrorBuffer<__half> output(N);
		CudaMirrorBuffer<__half> ref(N);

		random_init_array(input.host(), N);
		input.to_device();

		dim3 num_threads;
		dim3 threads_per_block;
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

		std::cout << "elementwise gelu\tversion " << version << "\t\tREF" << std::endl;
		std::cout << "Memory Bandwidth:\t" << elementwise_gelu::get_bytes_transferred(N) / 1e6 / time << " GB/s\t\t" << elementwise_gelu::get_bytes_transferred(N) / 1e6 / time_ref << " GB/s" << std::endl;
		std::cout << "Achieved GFLOPS:\t" << elementwise_gelu::get_FLOPs(N) / 1e6 / time << " GFLOPS(FP16)\t" << elementwise_gelu::get_FLOPs(N) / 1e6 / time_ref << " GFLOPS(FP16)" << std::endl;
		std::cout << std::endl;
	}
}

namespace stream_schedule
{
	void run(unsigned int version)
	{
		CudaMirrorBuffer<float> a(N);
		CudaMirrorBuffer<float> b(N);
		CudaMirrorBuffer<float> c(N);

		random_init_array(a.host(), N);
		random_init_array(b.host(), N);

		std::vector<cudaStream_t> streams(MAXNUMSTREAMS);
		for (auto& stream : streams)
			cudaStreamCreate(&stream);

		for (int num_streams = 1; num_streams <= MAXNUMSTREAMS; num_streams++)
			for (size_t i = 0; i < NREPEATS; i++)
				stream_schedule::kernels[version](streams.data(), num_streams, a.host(), b.host(), c.host(), a.device(), b.device(), c.device(), N);

		for (auto& stream : streams)
			cudaStreamDestroy(stream);
	}

	void test(unsigned int version)
	{
		CudaMirrorBuffer<float> a(N);
		CudaMirrorBuffer<float> b(N);
		CudaMirrorBuffer<float> c(N);
		CudaMirrorBuffer<float> ref(N);

		random_init_array(a.host(), N);
		random_init_array(b.host(), N);
		float* a_host = a.host();
		float* b_host = b.host();
		float* ref_host = ref.host();
		for (int j = 0; j < N; j++)
			ref_host[j] = a_host[j] + b_host[j];

		float milliseconds = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		std::vector<cudaStream_t> streams(MAXNUMSTREAMS);
		for (auto& stream : streams)
			cudaStreamCreate(&stream);

		for (int num_streams = 1; num_streams <= MAXNUMSTREAMS; num_streams++)
		{
			c.memset(0);

			for (size_t i = 0; i < WARMUP; i++)
				stream_schedule::kernels[version](streams.data(), num_streams, a.host(), b.host(), c.host(), a.device(), b.device(), c.device(), N);

			cudaEventRecord(start);

			for (size_t i = 0; i < NREPEATS; i++)
				stream_schedule::kernels[version](streams.data(), num_streams, a.host(), b.host(), c.host(), a.device(), b.device(), c.device(), N);

			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);

			float time = milliseconds / NREPEATS;

			compare_array(c.host(), ref.host(), N, 0.f);

			std::cout << "stream schedule\tversion " << version << "\t" << num_streams << " streams" << std::endl;
			std::cout << "Time cost:\t\t\t" << time << " ms\t" << std::endl;
			std::cout << std::endl;
		}

		for (auto& stream : streams)
			cudaStreamDestroy(stream);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
}

namespace quantize
{
	void quantize_cpu(float* h_input, int8_t* h_output)
	{
		float row_max;
		float row_min;
		float scale;
		float zeropoint;
		for (int j = 0; j < ROWS; j++)
		{
			row_max = FLT_MIN;
			row_min = FLT_MAX;
			for (int k = 0; k < COLS; k++)
			{
				row_max = max(row_max, h_input[j * COLS + k]);
				row_min = min(row_min, h_input[j * COLS + k]);
			}
			if constexpr (QUANTIZETYPE == ASYMMETRIC)
			{
				scale = (row_max - row_min) / (QMAX - QMIN);
				zeropoint = QMIN - nearbyint(row_min / scale);
			}
			else
			{
				scale = max(fabs(row_max), fabs(row_min)) / QMAX;
				zeropoint = 0.f;
			}
			for (int k = 0; k < COLS; k++)
				h_output[j * COLS + k] = int8_t(std::clamp(nearbyint(h_input[j * COLS + k] / scale + zeropoint), QMIN, QMAX));
		}
	}

	void run(unsigned int version)
	{
		CudaMirrorBuffer<float> input(ROWS * COLS);
		CudaMirrorBuffer<int8_t> output(ROWS * COLS);

		random_init_array(input.host(), ROWS * COLS);
		input.to_device();

		if constexpr (PROFILEREF)
		{
			for (int i = 0; i < NREPEATS; i++)
				quantize_cpu(input.host(), output.host());
		} 
		else
		{
			dim3 num_threads;
			dim3 threads_per_block;
			int shared_mem_bytes;
			quantize::get_kernel_launch_params(ROWS, COLS, version, num_threads, threads_per_block, shared_mem_bytes);
			for (size_t i = 0; i < NREPEATS; i++)
			{
				CUDA_LAUNCH_SHAREDMEM(quantize::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), output.device(), ROWS, COLS, QMIN, QMAX);
				CHECK_CUDA_ERROR("run kernel failed");
			}
		}
	}

	void test(unsigned int version)
	{
		CudaMirrorBuffer<float> input(ROWS * COLS);
		CudaMirrorBuffer<int8_t> output(ROWS * COLS);
		CudaMirrorBuffer<int8_t> ref(ROWS * COLS);

		random_init_array(input.host(), ROWS * COLS);
		input.to_device();

		dim3 num_threads;
		dim3 threads_per_block;
		int shared_mem_bytes;
		quantize::get_kernel_launch_params(ROWS, COLS, version, num_threads, threads_per_block, shared_mem_bytes);

		for (size_t i = 0; i < WARMUP; i++)
		{
			CUDA_LAUNCH_SHAREDMEM(quantize::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), output.device(), ROWS, COLS, QMIN, QMAX);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		float milliseconds = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		for (size_t i = 0; i < NREPEATS; i++)
		{
			CUDA_LAUNCH_SHAREDMEM(quantize::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), output.device(), ROWS, COLS, QMIN, QMAX);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		output.to_host();
		float time = milliseconds / NREPEATS;

		for (int i = 0; i < WARMUP; i++)
			quantize_cpu(input.host(), ref.host());

		auto begin = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < NREPEATS; i++)
			quantize_cpu(input.host(), ref.host());

		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - begin;

		double time_ref = elapsed.count() / NREPEATS * 1e3;

		compare_array(output.host(), ref.host(), ROWS * COLS, 0.f);

		std::cout << "quantize\t\tversion " << version << "\tREF" << std::endl;
		std::cout << "Memory Bandwidth:\t" << quantize::get_bytes_transferred(ROWS, COLS) / 1e6 / time << " GB/s\t" << quantize::get_bytes_transferred(ROWS, COLS) / 1e6 / time_ref << " GB/s" << std::endl;
		std::cout << "Achieved GFLOPS:\t" << quantize::get_FLOPs(ROWS, COLS) / 1e6 / time << " GFLOPS\t" << quantize::get_FLOPs(ROWS, COLS) / 1e6 / time_ref << " GFLOPS" << std::endl;
		std::cout << std::endl;
	}
}

namespace softmax
{
	void softmax_cpu(float* h_input, float* h_output)
	{
		float row_max;
		float row_exp_sum;
		for (int j = 0; j < ROWS; j++)
		{
			row_max = FLT_MIN;
			for (int k = 0; k < COLS; k++)
				row_max = max(row_max, h_input[j * COLS + k]);
			for (int k = 0; k < COLS; k++)
				h_output[j * COLS + k] = expf(h_input[j * COLS + k] - row_max);
			row_exp_sum = 0.0f;
			for (int k = 0; k < COLS; k++)
				row_exp_sum += h_output[j * COLS + k];
			for (int k = 0; k < COLS; k++)
				h_output[j * COLS + k] /= row_exp_sum;
		}
	}

	void run(unsigned int version)
	{
		CudaMirrorBuffer<float> input(ROWS * COLS);
		CudaMirrorBuffer<float> output(ROWS * COLS);

		random_init_array(input.host(), ROWS * COLS);
		input.to_device();

		if constexpr (PROFILEREF)
		{
			for (int i = 0; i < NREPEATS; i++)
				softmax_cpu(input.host(), output.host());
		}
		else
		{
			dim3 num_threads;
			dim3 threads_per_block;
			int shared_mem_bytes;
			softmax::get_kernel_launch_params(ROWS, COLS, version, num_threads, threads_per_block, shared_mem_bytes);
			for (size_t i = 0; i < NREPEATS; i++)
			{
				CUDA_LAUNCH_SHAREDMEM(softmax::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), output.device(), ROWS, COLS);
				CHECK_CUDA_ERROR("run kernel failed");
			}
		}
	}

	void test(unsigned int version)
	{
		CudaMirrorBuffer<float> input(ROWS * COLS);
		CudaMirrorBuffer<float> output(ROWS * COLS);
		CudaMirrorBuffer<float> ref(ROWS * COLS);

		random_init_array(input.host(), ROWS * COLS);
		input.to_device();

		dim3 num_threads;
		dim3 threads_per_block;
		int shared_mem_bytes;
		softmax::get_kernel_launch_params(ROWS, COLS, version, num_threads, threads_per_block, shared_mem_bytes);

		for (size_t i = 0; i < WARMUP; i++)
		{
			CUDA_LAUNCH_SHAREDMEM(softmax::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), output.device(), ROWS, COLS);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		float milliseconds = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		for (size_t i = 0; i < NREPEATS; i++)
		{
			CUDA_LAUNCH_SHAREDMEM(softmax::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(input.device(), output.device(), ROWS, COLS);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		output.to_host();
		float time = milliseconds / NREPEATS;

		for (int i = 0; i < WARMUP; i++)
			softmax_cpu(input.host(), ref.host());

		auto begin = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < NREPEATS; i++)
			softmax_cpu(input.host(), ref.host());

		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - begin;

		double time_ref = elapsed.count() / NREPEATS * 1e3;

		compare_array(output.host(), ref.host(), ROWS * COLS, TOLERANCETIGHT);

		std::cout << "softmax\t\t\tversion " << version << "\tREF" << std::endl;
		std::cout << "Memory Bandwidth:\t" << softmax::get_bytes_transferred(ROWS, COLS) / 1e6 / time << " GB/s\t" << softmax::get_bytes_transferred(ROWS, COLS) / 1e6 / time_ref << " GB/s" << std::endl;
		std::cout << "Achieved GFLOPS:\t" << softmax::get_FLOPs(ROWS, COLS) / 1e6 / time << " GFLOPS\t" << softmax::get_FLOPs(ROWS, COLS) / 1e6 / time_ref << " GFLOPS" << std::endl;
		std::cout << std::endl;
	}
}

namespace gemv_col_major
{
	void gemv_col_major_cpu(float* m, float* v, float* output)
	{
		std::fill(output, output + ROWS, 0.0f);
		for (size_t col_idx = 0; col_idx < COLS; col_idx++)
			for (size_t row_idx = 0; row_idx < ROWS; row_idx++)
				output[row_idx] += m[col_idx * ROWS + row_idx] * v[col_idx];
	}

	void run(unsigned int version) {}

	void test(unsigned int version)
	{
		CudaMirrorBuffer<float> m(ROWS * COLS);
		CudaMirrorBuffer<float> v(COLS);
		CudaMirrorBuffer<float> output(ROWS);
		CudaMirrorBuffer<float> ref(ROWS);

		random_init_array(m.host(), ROWS * COLS);
		random_init_array(v.host(), COLS);
		m.to_device();
		v.to_device();

		dim3 num_threads;
		dim3 threads_per_block;
		int shared_mem_bytes;
		gemv_col_major::get_kernel_launch_params(ROWS, COLS, version, num_threads, threads_per_block, shared_mem_bytes);

		for (size_t i = 0; i < WARMUP; i++)
		{
			CUDA_LAUNCH_SHAREDMEM(gemv_col_major::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(m.device(), v.device(), output.device(), ROWS, COLS);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		float milliseconds = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start);

		for (size_t i = 0; i < NREPEATS; i++)
		{
			CUDA_LAUNCH_SHAREDMEM(gemv_col_major::kernels[version], num_threads, threads_per_block, shared_mem_bytes)(m.device(), v.device(), output.device(), ROWS, COLS);
			CHECK_CUDA_ERROR("run kernel failed");
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&milliseconds, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		output.to_host();
		float time = milliseconds / NREPEATS;

		for (int i = 0; i < WARMUP; i++)
			gemv_col_major_cpu(m.host(), v.host(), ref.host());

		auto begin = std::chrono::high_resolution_clock::now();

		for (int i = 0; i < NREPEATS; i++)
			gemv_col_major_cpu(m.host(), v.host(), ref.host());

		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - begin;

		double time_ref = elapsed.count() / NREPEATS * 1e3;

		compare_array(output.host(), ref.host(), ROWS, TOLERANCETIGHT);

		std::cout << "gemv col major\t\tversion " << version << "\tREF" << std::endl;
		std::cout << "Memory Bandwidth:\t" << gemv_col_major::get_bytes_transferred(ROWS, COLS) / 1e6 / time << " GB/s\t" << gemv_col_major::get_bytes_transferred(ROWS, COLS) / 1e6 / time_ref << " GB/s" << std::endl;
		std::cout << "Achieved GFLOPS:\t" << gemv_col_major::get_FLOPs(ROWS, COLS) / 1e6 / time << " GFLOPS\t" << gemv_col_major::get_FLOPs(ROWS, COLS) / 1e6 / time_ref << " GFLOPS" << std::endl;
		std::cout << std::endl;
	}
}