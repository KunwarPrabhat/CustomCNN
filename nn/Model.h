#pragma once
#include <vector>
#include "../Layers/Layer.h"
#include "../core/Tensor.h"

class Model {
private:
    std::vector<Layer*> layers;

public:
    void add(Layer* layer);
    Tensor forward(const Tensor& input);
};