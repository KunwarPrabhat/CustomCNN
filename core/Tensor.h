#pragma once
#include <vector>

class Tensor {
public:
  std::vector<float> data;
  int rows, cols;

  Tensor(int r, int c);

  float &operator()(int i, int j);
  float operator()(int i, int j) const;
  void randomize();
};