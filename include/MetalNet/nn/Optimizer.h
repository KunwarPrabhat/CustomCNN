#pragma once
#include <vector>
#include <memory>
#include "../Layers/Layer.h"

namespace MetalNet {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step(std::vector<std::shared_ptr<Layer>>& layers) = 0;
};

class SGD : public Optimizer {
public:
    float lr;
    inline SGD(float l=0.01f) : lr(l) {}
    inline void step(std::vector<std::shared_ptr<Layer>>& layers) override {
        for (auto& l : layers) l->update_weights(lr);
    }
};

} // namespace MetalNet
