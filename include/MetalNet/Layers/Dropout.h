#pragma once
#include "Layer.h"
#include <cstdlib>

namespace MetalNet {

class Dropout : public Layer {
public:
    float  rate;
    Tensor mask;
    inline Dropout(float r=0.5f) : rate(r) {}

    inline Tensor forward(const Tensor& input) override {
        if (!is_training) return input;
        mask = Tensor(input.shape);
        Tensor out(input.shape);
        const float sc=1.0f/(1.0f-rate);
        float* m=mask.data.data(); float* o=out.data.data(); const float* s=input.data.data();
        for (int i=0;i<input.size();++i) {
            float r=(float)rand()/RAND_MAX;
            m[i]=r>rate?1.0f:0.0f; o[i]=m[i]*s[i]*sc;
        }
        return out;
    }
    inline Tensor backward(const Tensor& go) override {
        if (!is_training) return go;
        const float sc=1.0f/(1.0f-rate);
        Tensor d(go.shape);
        const float* g=go.data.data(); const float* m=mask.data.data(); float* di=d.data.data();
        #pragma omp simd
        for (int i=0;i<go.size();++i) di[i]=g[i]*m[i]*sc;
        return d;
    }
};

} // namespace MetalNet
