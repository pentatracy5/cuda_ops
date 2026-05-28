#include <iostream>
#include <sstream>
#include <test.cuh>
#include <config.cuh>

void run(unsigned int type, unsigned int version)
{
	if (0 == type)
		elementwise_add::run(version);
	else if (1 == type)
		reduce_sum::run(version);
	else if (2 == type)
		histogram::run(version);
	else if (3 == type)
		copy_if::run(version);
	else if (4 == type)
		elementwise_gelu::run(version);
	else if (5 == type)
		stream_schedule::run(version);
	else if (6 == type)
		quantize::run(version);
	else if (7 == type)
		softmax::run(version);
	else if (8 == type)
		gemv_col_major::run(version);
	else if (9 == type)
		gemv_row_major::run(version);
	else if (10 == type)
		elementwise_dropout::run(version);
}

void test(unsigned int type, unsigned int version)
{
	if (0 == type)
		elementwise_add::test(version);
	else if (1 == type)
		reduce_sum::test(version);
	else if (2 == type)
		histogram::test(version);
	else if (3 == type)
		copy_if::test(version);
	else if (4 == type)
		elementwise_gelu::test(version);
	else if (5 == type)
		stream_schedule::test(version);
	else if (6 == type)
		quantize::test(version);
	else if (7 == type)
		softmax::test(version);
	else if (8 == type)
		gemv_col_major::test(version);
	else if (9 == type)
		gemv_row_major::test(version);
	else if (10 == type)
		elementwise_dropout::test(version);
}

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		std::cout << "Error: require 2 arguments, but " << argc - 1 << " provided." << std::endl;
		return 1;
	}

	unsigned int type;
	std::istringstream iss1(argv[1]);
	if (!(iss1 >> type)) {
		std::cerr << "Error: invalid ops type." << std::endl;
		return 1;
	}

	unsigned int version;
	std::istringstream iss2(argv[2]);
	if (!(iss2 >> version)) {
		std::cerr << "Error: invalid ops version." << std::endl;
		return 1;
	}

	if constexpr (PROFILE)
		run(type, version);
	else
		test(type, version);

	return 0;
}