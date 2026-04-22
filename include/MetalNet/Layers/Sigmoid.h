#pragma once
#include "Layer.h"
#include <cmath>

namespace MetalNet {

class Sigmoid : public Layer {
public:
    inline Tensor forward(const Tensor& input) override {
        cached_input = input;
        Tensor out(input.shape);
        const float* s = input.data.data(); float* d = out.data.data();
        #pragma omp simd
        for (int i=0;i<input.size();++i) d[i] = 1.0f / (1.0f + std::exp(-s[i]));
        output_cache = out;
        return out;
    }
    inline Tensor backward(const Tensor& go) override {
        Tensor d(go.shape);
        const float* g = go.data.data(); 
        const float* out_val = output_cache.data.data(); 
        float* di = d.data.data();
        #pragma omp simd
        for (int i=0;i<go.size();++i) di[i] = g[i] * out_val[i] * (1.0f - out_val[i]);
        return d;
    }
};

} // namespace MetalNet
