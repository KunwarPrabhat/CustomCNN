#pragma once
#include "Layer.h"
#include <limits>

namespace MetalNet {

class MaxPool2D : public Layer {
public:
    int pool_size, stride;
    inline MaxPool2D(int p, int s=2) : pool_size(p), stride(s) {}

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        const int N = input_shapes[0][0], C = input_shapes[0][1];
        const int H = input_shapes[0][2], W = input_shapes[0][3];
        const int OH = (H - pool_size) / stride + 1, OW = (W - pool_size) / stride + 1;
        output_buffer = Tensor(N, C, OH, OW);
        grad_input_buffer = Tensor(input_shapes[0]);
    }

    inline void forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const int N=input.shape[0],C=input.shape[1],H=input.shape[2],W=input.shape[3];
        const int OH=output_buffer.shape[2], OW=output_buffer.shape[3];
        const float* src = input.data.data();
        float*       dst = output_buffer.data.data();
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
    }

    inline void backward(const Tensor& go) override {
        const int N=cached_input_ptr->shape[0],C=cached_input_ptr->shape[1];
        const int H=cached_input_ptr->shape[2],W=cached_input_ptr->shape[3];
        const int OH=go.shape[2], OW=go.shape[3];
        grad_input_buffer.fill(0.0f);
        const float* src = cached_input_ptr->data.data();
        const float* g   = go.data.data();
        float*       di  = grad_input_buffer.data.data();
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
    }
};

} // namespace MetalNet
