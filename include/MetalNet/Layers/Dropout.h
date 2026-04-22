#pragma once
#include "Layer.h"
#include <cstdlib>

namespace MetalNet {

class Dropout : public Layer {
public:
    float  rate;
    Tensor mask;
    inline Dropout(float r=0.5f) : rate(r) {}

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        output_buffer = Tensor(input_shapes[0]);
        grad_input_buffer = Tensor(input_shapes[0]);
        mask = Tensor(input_shapes[0]);
    }

    inline void forward(const Tensor& input) override {
        if (!is_training) {
            const float* s=input.data.data(); float* d=output_buffer.data.data();
            #pragma omp simd
            for(int i=0;i<input.size();++i) d[i]=s[i];
            return;
        }
        const float sc=1.0f/(1.0f-rate);
        float* m=mask.data.data(); float* o=output_buffer.data.data(); const float* s=input.data.data();
        for (int i=0;i<input.size();++i) {
            float r=(float)rand()/RAND_MAX;
            m[i]=r>rate?1.0f:0.0f; o[i]=m[i]*s[i]*sc;
        }
    }
    inline void backward(const Tensor& go) override {
        if (!is_training) {
            const float* g=go.data.data(); float* di=grad_input_buffer.data.data();
            #pragma omp simd
            for(int i=0;i<go.size();++i) di[i]=g[i];
            return;
        }
        const float sc=1.0f/(1.0f-rate);
        const float* g=go.data.data(); const float* m=mask.data.data(); float* di=grad_input_buffer.data.data();
        #pragma omp simd
        for (int i=0;i<go.size();++i) di[i]=g[i]*m[i]*sc;
    }
};

} // namespace MetalNet
