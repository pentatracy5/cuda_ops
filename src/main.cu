#include <iostream>
#include <sstream>
#include <test.cuh>
#include <config.cuh>

using std::cout;
using std::endl; 
using std::istringstream;
using std::cerr;

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