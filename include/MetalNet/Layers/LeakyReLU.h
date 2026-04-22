#pragma once
#include "Layer.h"

namespace MetalNet {

class LeakyReLU : public Layer {
public:
    float negative_slope;
    inline LeakyReLU(float a=0.01f) : negative_slope(a) {}

    inline Tensor forward(const Tensor& input) override {
        cached_input = input;
        Tensor out(input.shape);
        const float* s=input.data.data(); float* d=out.data.data(); float ns=negative_slope;
        #pragma omp simd
        for (int i=0;i<input.size();++i) d[i] = s[i]>0?s[i]:s[i]*ns;
        return out;
    }
    inline Tensor backward(const Tensor& go) override {
        Tensor d(go.shape);
        const float* g=go.data.data(); const float* ci=cached_input.data.data();
        float* di=d.data.data(); float ns=negative_slope;
        #pragma omp simd
        for (int i=0;i<go.size();++i) di[i] = ci[i]>0?g[i]:g[i]*ns;
        return d;
    }
};

} // namespace MetalNet
