#pragma once
#include "../core/Tensor.h"

class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
};