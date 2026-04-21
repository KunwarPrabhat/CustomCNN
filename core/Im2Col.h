#pragma once
#include "Tensor.h"

Tensor im2col(const Tensor& input, int kernel_size, int stride = 1, int padding = 0);
Tensor col2im(const Tensor& col_tensor, const std::vector<int>& input_shape, int kernel_size, int stride = 1, int padding = 0);
