#pragma once
#include "Layer.h"

class Dense : public Layer {
private:
    Tensor weights;
    Tensor bias;

public:
    Dense(int input_size, int output_size);
    Tensor forward(const Tensor& input) override;
};