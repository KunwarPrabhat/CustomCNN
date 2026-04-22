#pragma once
#include "Layer.h"
#include <limits>

namespace MetalNet {

class MaxPool2D : public Layer {
public:
    int pool_size, stride;
    inline MaxPool2D(int p, int s=2) : pool_size(p), stride(s) {}

    inline Tensor forward(const Tensor& input) override {
        cached_input = input;
        const int N=input.shape[0],C=input.shape[1],H=input.shape[2],W=input.shape[3];
        const int OH=(H-pool_size)/stride+1, OW=(W-pool_size)/stride+1;
        Tensor output(N,C,OH,OW);
        const float* src = input.data.data();
        float*       dst = output.data.data();
        #pragma omp parallel for collapse(2) schedule(static)
        for (int n=0;n<N;++n) for (int c=0;c<C;++c) {
            const float* sc = src + n*(C*H*W) + c*(H*W);
            float*       dc = dst + n*(C*OH*OW) + c*(OH*OW);
            for (int y=0;y<OH;++y) for (int x=0;x<OW;++x) {
                float mv = -std::numeric_limits<float>::infinity();
                for (int py=0;py<pool_size;++py) for (int px=0;px<pool_size;++px) {
                    float v = sc[(y*stride+py)*W + x*stride+px];
                    if (v>mv) mv=v;
                }
                dc[y*OW+x] = mv;
            }
        }
        return output;
    }

    inline Tensor backward(const Tensor& go) override {
        const int N=cached_input.shape[0],C=cached_input.shape[1];
        const int H=cached_input.shape[2],W=cached_input.shape[3];
        const int OH=go.shape[2], OW=go.shape[3];
        Tensor d_in(cached_input.shape); d_in.fill(0.0f);
        const float* src = cached_input.data.data();
        const float* g   = go.data.data();
        float*       di  = d_in.data.data();
        #pragma omp parallel for collapse(2) schedule(static)
        for (int n=0;n<N;++n) for (int c=0;c<C;++c) {
            const float* sc = src + n*(C*H*W) + c*(H*W);
            const float* gc = g   + n*(C*OH*OW) + c*(OH*OW);
            float*       dc = di  + n*(C*H*W) + c*(H*W);
            for (int y=0;y<OH;++y) for (int x=0;x<OW;++x) {
                float mv=-std::numeric_limits<float>::infinity(); int my=0,mx=0;
                for (int py=0;py<pool_size;++py) for (int px=0;px<pool_size;++px) {
                    int iy=y*stride+py, ix=x*stride+px;
                    float v=sc[iy*W+ix]; if(v>mv){mv=v;my=iy;mx=ix;}
                }
                dc[my*W+mx] += gc[y*OW+x];
            }
        }
        return d_in;
    }
};

} // namespace MetalNet
