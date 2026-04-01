#pragma once
#include "Layer.h"

class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override;
};