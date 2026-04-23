#pragma once
#include "Layer.h"
#include <stdexcept>

namespace MetalNet {

class Concat : public Layer {
public:
    std::vector<int> channel_splits;

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        if (input_shapes.empty()) throw std::runtime_error("Concat: needs >0 inputs");
        int total_c=0, N=input_shapes[0][0], H=input_shapes[0][2], W=input_shapes[0][3];
        channel_splits.clear();
        multi_grad_input_buffers.clear();
        for (const auto& sh: input_shapes) { 
            total_c += sh[1]; 
            channel_splits.push_back(sh[1]); 
            multi_grad_input_buffers.push_back(Tensor(sh));
        }
        output_buffer = Tensor(N,total_c,H,W);
    }

    inline Tensor& forward(const std::vector<const Tensor*>& inputs) override {
        if (inputs.empty()) throw std::runtime_error("Concat: needs >0 inputs");
        int total_c=0, N=inputs[0]->shape[0], H=inputs[0]->shape[2], W=inputs[0]->shape[3];
        for (int sp : channel_splits) total_c += sp;
        float* dst = output_buffer.data.data();
        
        for (int n=0;n<N;++n) {
            int cc=0;
            for (size_t i=0;i<inputs.size();++i) {
                int ci=channel_splits[i];
                const float* src=inputs[i]->data.data();
                for (int c=0;c<ci;++c) {
                    std::span<const float> sv(src+n*(ci*H*W)+c*(H*W), H*W);
                    std::span<float>       dv(dst+n*(total_c*H*W)+(cc+c)*(H*W), H*W);
                    std::copy(sv.begin(), sv.end(), dv.begin());
                }
                cc+=ci;
            }
        }
        return output_buffer;
    }

    inline void backward_multi(const Tensor& go) override {
        const int N=go.shape[0], H=go.shape[2], W=go.shape[3];
        const int total_c=go.shape[1];
        const float* src=go.data.data();
        
        #pragma omp parallel for schedule(static)
        for (int n=0;n<N;++n) {
            int cc=0;
            for (size_t i=0; i<channel_splits.size(); ++i) {
                int sp = channel_splits[i];
                float* d=multi_grad_input_buffers[i].data.data();
                for (int c=0;c<sp;++c) {
                    std::span<const float> sv(src+n*(total_c*H*W)+(cc+c)*(H*W), H*W);
                    std::span<float>       dv(d  +n*(sp*H*W)    +c       *(H*W), H*W);
                    std::copy(sv.begin(), sv.end(), dv.begin());
                }
                cc+=sp;
            }
        }
    }
    inline std::string name() const override { return "Concat"; }
};

} // namespace MetalNet
