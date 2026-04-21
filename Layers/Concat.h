#pragma once
#include "Layer.h"
#include <vector>

class Concat : public Layer {
public:
    Tensor forward(const std::vector<Tensor>& inputs) override;
    std::vector<Tensor> backward_multi(const Tensor& grad_output) override;
private:
    std::vector<int> channel_splits;
};
