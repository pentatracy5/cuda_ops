#pragma once

namespace elementwise_add 
{
	void run(unsigned int version);
	void test(unsigned int version);
}

namespace reduce_sum
{
	void run(unsigned int version);
	void test(unsigned int version);
}

namespace histogram
{
	void run(unsigned int version);
	void test(unsigned int version);
}

namespace copy_if
{
	void run(unsigned int version);
	void test(unsigned int version);
}

namespace elementwise_gelu
{
	void run(unsigned int version);
	void test(unsigned int version);
}

namespace stream_schedule
{
	void run(unsigned int version);
	void test(unsigned int version);
}

namespace quantize
{
	void run(unsigned int version);
	void test(unsigned int version);
}

namespace softmax
{
	void run(unsigned int version);
	void test(unsigned int version);
}

namespace gemv_col_major
{
	void run(unsigned int version);
	void test(unsigned int version);
}