#pragma once
#include "Layer.h"
#include <stdexcept>

namespace MetalNet {

class ElementwiseAdd : public Layer {
public:
    inline Tensor forward(const std::vector<Tensor>& inputs) override {
        if (inputs.size()<2) throw std::runtime_error("ElementwiseAdd: needs >=2 inputs");
        Tensor out = inputs[0];
        float* d = out.data.data();
        for (size_t i=1;i<inputs.size();++i) {
            const float* s=inputs[i].data.data();
            #pragma omp simd
            for (int j=0;j<out.size();++j) d[j]+=s[j];
        }
        return out;
    }
    inline std::vector<Tensor> backward_multi(const Tensor& go) override {
        std::vector<Tensor> grads;
        for (size_t i=0;i<input_nodes.size();++i) grads.push_back(go);
        return grads;
    }
};

} // namespace MetalNet
