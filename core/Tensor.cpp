#include "Tensor.h"
#include <cstdlib>

Tensor::Tensor(int r, int c) : rows(r), cols(c) { data.resize(r * c); }

float &Tensor::operator()(int i, int j) { return data[i * cols + j]; }

float Tensor::operator()(int i, int j) const { return data[i * cols + j]; }

void Tensor::randomize() {
  for (auto &x : data)
    x = ((float)rand() / RAND_MAX) * 2 - 1; // [-1,1]
}
