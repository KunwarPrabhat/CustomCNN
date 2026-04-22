#pragma once
#include "Layer.h"
#include <stdexcept>

namespace MetalNet {

class Concat : public Layer {
public:
    std::vector<int> channel_splits;

    inline Tensor forward(const std::vector<Tensor>& inputs) override {
        if (inputs.empty()) throw std::runtime_error("Concat: needs >0 inputs");
        int total_c=0, N=inputs[0].shape[0], H=inputs[0].shape[2], W=inputs[0].shape[3];
        channel_splits.clear();
        for (const auto& t:inputs) { total_c+=t.shape[1]; channel_splits.push_back(t.shape[1]); }
        Tensor out(N,total_c,H,W);
        float* dst=out.data.data();
        for (int n=0;n<N;++n) {
            int cc=0;
            for (size_t i=0;i<inputs.size();++i) {
                int ci=inputs[i].shape[1];
                const float* src=inputs[i].data.data();
                for (int c=0;c<ci;++c) {
                    std::span<const float> sv(src+n*(ci*H*W)+c*(H*W), H*W);
                    std::span<float>       dv(dst+n*(total_c*H*W)+(cc+c)*(H*W), H*W);
                    std::copy(sv.begin(), sv.end(), dv.begin());
                }
                cc+=ci;
            }
        }
        return out;
    }

    inline std::vector<Tensor> backward_multi(const Tensor& go) override {
        std::vector<Tensor> grads;
        const int N=go.shape[0], H=go.shape[2], W=go.shape[3];
        const int total_c=go.shape[1];
        const float* src=go.data.data(); int cc=0;
        for (int sp:channel_splits) {
            Tensor g(N,sp,H,W); float* d=g.data.data();
            for (int n=0;n<N;++n) for (int c=0;c<sp;++c) {
                std::span<const float> sv(src+n*(total_c*H*W)+(cc+c)*(H*W), H*W);
                std::span<float>       dv(d  +n*(sp*H*W)    +c       *(H*W), H*W);
                std::copy(sv.begin(), sv.end(), dv.begin());
            }
            cc+=sp; grads.push_back(g);
        }
        return grads;
    }
};

} // namespace MetalNet
