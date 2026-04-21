#pragma once
#include <vector>
#include <memory>
#include "../Layers/Layer.h"

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(std::vector<std::shared_ptr<Layer>>& layers) = 0;
};

class SGD : public Optimizer {
public:
    float learning_rate;

    SGD(float lr = 0.01f);
    void step(std::vector<std::shared_ptr<Layer>>& layers) override;
};
