#pragma once
#include "Optimizer.h"
#include <unordered_map>
#include <cmath>

class Adam : public Optimizer {
public:
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    int t;

    std::unordered_map<Tensor*, Tensor> m;
    std::unordered_map<Tensor*, Tensor> v;

    Adam(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f);

    void step(std::vector<std::shared_ptr<Layer>>& layers) override;
};
