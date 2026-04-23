#include <utils.cuh>
#include <random>
#include <algorithm>
#include <iostream>

using std::fill;
using std::mt19937;
using std::random_device;
using std::uniform_real_distribution;
using std::cout;
using std::endl;

void constant_val_init_array(float* array, const int size, const float val)
{
	fill(array, array + size, val);
}

void random_init_array(float* array, const int size)
{
	mt19937 engine(random_device{}());
	uniform_real_distribution<float> dist(0.0f, 1.0f);
	for (int i = 0; i < size; i++)
		array[i] = dist(engine);
}

void compare_array(float* c, float* ref, const int size, const float tolerance)
{
	for (int i = 0; i < size; i++)
		if (fabs(c[i] - ref[i]) > tolerance)
		{
			cout << "Error: c(" << i << ") = " << c[i] << ", but expected " << ref[i] << endl;
			return;
		}
	return;
}