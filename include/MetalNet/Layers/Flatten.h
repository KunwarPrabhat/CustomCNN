#pragma once
#include "Layer.h"

namespace MetalNet {

class Flatten : public Layer {
public:
    inline Tensor forward(const Tensor& input) override {
        cached_input = input;
        int N=input.shape[0], fd=1;
        for (int i=1;i<(int)input.shape.size();++i) fd*=input.shape[i];
        Tensor out(std::vector<int>{N,fd});
        out.data = input.data;
        return out;
    }
    inline Tensor backward(const Tensor& go) override {
        Tensor d(cached_input.shape);
        d.data = go.data;
        return d;
    }
};

} // namespace MetalNet
