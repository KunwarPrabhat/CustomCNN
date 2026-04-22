#pragma once
#include "Layer.h"
#include <cmath>

namespace MetalNet {

class Sigmoid : public Layer {
public:
    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        output_buffer = Tensor(input_shapes[0]);
        grad_input_buffer = Tensor(input_shapes[0]);
    }

    inline void forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const float* s = input.data.data(); float* d = output_buffer.data.data();
        #pragma omp simd
        for (int i=0;i<input.size();++i) d[i] = 1.0f / (1.0f + std::exp(-s[i]));
    }
    inline void backward(const Tensor& go) override {
        const float* g = go.data.data(); 
        const float* out_val = output_buffer.data.data(); 
        float* di = grad_input_buffer.data.data();
        #pragma omp simd
        for (int i=0;i<go.size();++i) di[i] = g[i] * out_val[i] * (1.0f - out_val[i]);
    }
};

} // namespace MetalNet
