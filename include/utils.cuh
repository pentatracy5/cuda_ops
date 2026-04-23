#pragma once

void constant_val_init_array(float* array, int size, const float val);

void random_init_array(float* array, int size);

void compare_array(float* c, float* ref, const int size, const float tolerance);