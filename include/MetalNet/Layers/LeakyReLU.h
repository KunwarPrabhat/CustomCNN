#pragma once
#include "Layer.h"

namespace MetalNet {

class LeakyReLU : public Layer {
public:
    float negative_slope;
    inline LeakyReLU(float a=0.01f) : negative_slope(a) {}

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        output_buffer = Tensor(input_shapes[0]);
        grad_input_buffer = Tensor(input_shapes[0]);
    }

    inline void forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const float* s=input.data.data(); float* d=output_buffer.data.data(); float ns=negative_slope;
        #pragma omp simd
        for (int i=0;i<input.size();++i) d[i] = s[i]>0?s[i]:s[i]*ns;
    }
    inline void backward(const Tensor& go) override {
        const float* g=go.data.data(); const float* ci=cached_input_ptr->data.data();
        float* di=grad_input_buffer.data.data(); float ns=negative_slope;
        #pragma omp simd
        for (int i=0;i<go.size();++i) di[i] = ci[i]>0?g[i]:g[i]*ns;
    }
};

} // namespace MetalNet
