#pragma once
#include "Layer.h"

namespace MetalNet {

class Conv2D : public Layer {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor weights, biases;

    inline Conv2D(int in_c, int out_c, int k, int s=1, int p=0)
        : in_channels(in_c), out_channels(out_c), kernel_size(k), stride(s), padding(p) {
        weights = Tensor(std::vector<int>{out_c, in_c, k, k});
        biases  = Tensor(out_c, 1);
        weights.randomize();
        biases.fill(0.1f);
    }

    inline Tensor forward(const Tensor& input) override {
        cached_input = input;
        const int N=input.shape[0], H=input.shape[2], W=input.shape[3];
        const int OH=(H+2*padding-kernel_size)/stride+1;
        const int OW=(W+2*padding-kernel_size)/stride+1;
        const int K2=kernel_size*kernel_size;
        Tensor output(N, out_channels, OH, OW);

        const float* inp = input.data.data();
        const float* wt  = weights.data.data();
        const float* bs  = biases.data.data();
        float*       out = output.data.data();

        #pragma omp parallel for collapse(2) schedule(static)
        for (int n = 0; n < N; ++n) {
            for (int oc = 0; oc < out_channels; ++oc) {
                const float* w_oc = wt + oc*(in_channels*K2);
                float*       o_nc = out + n*(out_channels*OH*OW) + oc*(OH*OW);
                for (int y = 0; y < OH; ++y) {
                    for (int x = 0; x < OW; ++x) {
                        float acc = bs[oc];
                        int ki = 0;
                        for (int ic = 0; ic < in_channels; ++ic) {
                            const float* i_nc = inp + n*(in_channels*H*W) + ic*(H*W);
                            for (int ky = 0; ky < kernel_size; ++ky) {
                                int iy = y*stride - padding + ky;
                                for (int kx = 0; kx < kernel_size; ++kx, ++ki) {
                                    int ix = x*stride - padding + kx;
                                    if (iy>=0 && iy<H && ix>=0 && ix<W)
                                        acc += i_nc[iy*W + ix] * w_oc[ki];
                                }
                            }
                        }
                        o_nc[y*OW + x] = acc;
                    }
                }
            }
        }
        return output;
    }

    inline Tensor backward(const Tensor& grad_output) override {
        const int N=cached_input.shape[0], H=cached_input.shape[2], W=cached_input.shape[3];
        const int OH=grad_output.shape[2], OW=grad_output.shape[3];
        const int K2=kernel_size*kernel_size;
        Tensor d_input(cached_input.shape);

        const float* inp = cached_input.data.data();
        const float* go  = grad_output.data.data();
        const float* wt  = weights.data.data();
        float*       di  = d_input.data.data();
        float*       dw  = weights.grad.data();
        float*       db  = biases.grad.data();

        #pragma omp parallel for collapse(2) schedule(static)
        for (int n = 0; n < N; ++n) {
            for (int oc = 0; oc < out_channels; ++oc) {
                const float* go_noc = go + n*(out_channels*OH*OW) + oc*(OH*OW);
                const float* w_oc   = wt + oc*(in_channels*K2);
                int ki = 0;
                for (int ic = 0; ic < in_channels; ++ic) {
                    const float* i_nc = inp + n*(in_channels*H*W) + ic*(H*W);
                    float*       d_nc = di  + n*(in_channels*H*W) + ic*(H*W);
                    for (int ky = 0; ky < kernel_size; ++ky) {
                        for (int kx = 0; kx < kernel_size; ++kx, ++ki) {
                            float dw_acc = 0.0f;
                            for (int y = 0; y < OH; ++y) {
                                int iy = y*stride - padding + ky;
                                for (int x = 0; x < OW; ++x) {
                                    int ix = x*stride - padding + kx;
                                    if (iy>=0&&iy<H&&ix>=0&&ix<W) {
                                        float g = go_noc[y*OW + x];
                                        dw_acc        += i_nc[iy*W + ix] * g;
                                        d_nc[iy*W+ix] += w_oc[ki] * g;
                                    }
                                }
                            }
                            #pragma omp atomic
                            dw[oc*(in_channels*K2) + ki - 1] += dw_acc;
                        }
                    }
                }
                float db_acc = 0.0f;
                for (int hw = 0; hw < OH*OW; ++hw) db_acc += go_noc[hw];
                #pragma omp atomic
                db[oc] += db_acc;
            }
        }
        return d_input;
    }

    inline void update_weights(float lr) override {
        float* W=weights.data.data(); float* dW=weights.grad.data();
        float* b=biases.data.data();  float* db=biases.grad.data();
        #pragma omp simd
        for (int i=0;i<weights.size();++i) W[i]-=lr*dW[i];
        #pragma omp simd
        for (int i=0;i<biases.size();++i)  b[i]-=lr*db[i];
        weights.zero_grad(); biases.zero_grad();
    }

    inline std::vector<Tensor*> get_parameters() override { return {&weights, &biases}; }
};

} // namespace MetalNet
