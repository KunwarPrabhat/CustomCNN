#pragma once
#include <vector>
#include "../layers/Layer.h"

class Model {
private:
    std::vector<Layer*> layers;

public:
    void add(Layer* layer);
    Tensor forward(const Tensor& input);
};