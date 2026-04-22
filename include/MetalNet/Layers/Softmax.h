#pragma once
#include "Layer.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace MetalNet {

class Softmax : public Layer {
public:
    inline Tensor forward(const Tensor& input) override {
        const int N=input.shape[0], C=input.shape[1];
        Tensor out(input.shape);
        const float* src=input.data.data(); float* dst=out.data.data();
        for (int n=0;n<N;++n) {
            std::span<const float> row(src+n*C, C);
            std::span<float>       orow(dst+n*C, C);
            float mx=-std::numeric_limits<float>::infinity();
            for (float v:row) mx=std::max(mx,v);
            float se=0; for (float v:row) se+=std::exp(v-mx);
            for (int j=0;j<C;++j) orow[j]=std::exp(row[j]-mx)/se;
        }
        output_cache = out;
        return out;
    }
    inline Tensor backward(const Tensor& go) override {
        const int N=go.shape[0], C=go.shape[1];
        Tensor d(go.shape);
        const float* g=go.data.data(); const float* oc=output_cache.data.data(); float* di=d.data.data();
        for (int n=0;n<N;++n) {
            std::span<const float> grow(g+n*C,C), ocrow(oc+n*C,C);
            std::span<float> dr(di+n*C,C);
            float dot=0; for (int j=0;j<C;++j) dot+=ocrow[j]*grow[j];
            for (int j=0;j<C;++j) dr[j]=ocrow[j]*(grow[j]-dot);
        }
        return d;
    }
};

} // namespace MetalNet
