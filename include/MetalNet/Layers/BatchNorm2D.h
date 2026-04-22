#pragma once
#include "Layer.h"
#include <cmath>
#include <span>

namespace MetalNet {

class BatchNorm2D : public Layer {
public:
    int   num_features;
    float momentum;
    float eps;
    Tensor gamma, beta;
    Tensor running_mean, running_var;
    Tensor x_norm, batch_var;

    inline BatchNorm2D(int features, float mom=0.1f, float e=1e-5f)
        : num_features(features), momentum(mom), eps(e) {
        gamma=Tensor(features,1); beta=Tensor(features,1);
        running_mean=Tensor(features,1); running_var=Tensor(features,1);
        gamma.fill(1.0f); beta.fill(0.0f);
        running_mean.fill(0.0f); running_var.fill(1.0f);
    }

    inline std::vector<Tensor*> get_parameters() override { return {&gamma, &beta}; }
    inline std::vector<Tensor*> get_states()     override {
        return {&gamma, &beta, &running_mean, &running_var};
    }

    inline Tensor forward(const Tensor& input) override {
        const int N=input.shape[0], C=input.shape[1];
        const int H=input.shape[2], W=input.shape[3];
        const int HW=H*W, m=N*HW;
        Tensor output(input.shape);

        const float* inp = input.data.data();
        float*       out = output.data.data();
        const float* gam = gamma.data.data();
        const float* bet = beta.data.data();

        if (is_training) {
            std::vector<float> mean(C, 0.0f), var(C, 0.0f);

            #pragma omp parallel for schedule(static)
            for (int c=0; c<C; ++c) {
                float acc=0.0f;
                for (int b=0; b<N; ++b) {
                    std::span<const float> row(inp + b*(C*HW) + c*HW, HW);
                    #pragma omp simd reduction(+:acc)
                    for (int hw=0; hw<HW; ++hw) acc += row[hw];
                }
                mean[c] = acc / m;
            }

            #pragma omp parallel for schedule(static)
            for (int c=0; c<C; ++c) {
                float acc=0.0f, mu=mean[c];
                for (int b=0; b<N; ++b) {
                    std::span<const float> row(inp + b*(C*HW) + c*HW, HW);
                    #pragma omp simd reduction(+:acc)
                    for (int hw=0; hw<HW; ++hw) { float d=row[hw]-mu; acc+=d*d; }
                }
                var[c] = acc / m;
            }

            batch_var = Tensor(C,1);
            for (int c=0; c<C; ++c) {
                running_mean.data[c] = (1-momentum)*running_mean.data[c] + momentum*mean[c];
                running_var.data[c]  = (1-momentum)*running_var.data[c]  + momentum*var[c];
                batch_var.data[c]    = var[c];
            }

            x_norm = Tensor(input.shape);
            float* xn = x_norm.data.data();

            #pragma omp parallel for schedule(static)
            for (int c=0; c<C; ++c) {
                const float inv_std=1.0f/std::sqrt(var[c]+eps);
                const float gc=gam[c], bc=bet[c], mu=mean[c];
                for (int b=0; b<N; ++b) {
                    std::span<const float> sv(inp + b*(C*HW) + c*HW, HW);
                    std::span<float>       dv(out + b*(C*HW) + c*HW, HW);
                    std::span<float>       xv(xn  + b*(C*HW) + c*HW, HW);
                    #pragma omp simd
                    for (int hw=0; hw<HW; ++hw) {
                        xv[hw] = (sv[hw] - mu) * inv_std;
                        dv[hw] = gc * xv[hw] + bc;
                    }
                }
            }
        } else {
            const float* rm = running_mean.data.data();
            const float* rv = running_var.data.data();
            #pragma omp parallel for schedule(static)
            for (int c=0; c<C; ++c) {
                const float inv_std=1.0f/std::sqrt(rv[c]+eps);
                const float gc=gam[c], bc=bet[c], mu=rm[c];
                for (int b=0; b<N; ++b) {
                    std::span<const float> sv(inp + b*(C*HW) + c*HW, HW);
                    std::span<float>       dv(out + b*(C*HW) + c*HW, HW);
                    #pragma omp simd
                    for (int hw=0; hw<HW; ++hw)
                        dv[hw] = gc * (sv[hw] - mu) * inv_std + bc;
                }
            }
        }
        return output;
    }

    inline Tensor backward(const Tensor& grad_output) override {
        const int N=grad_output.shape[0], C=grad_output.shape[1];
        const int H=grad_output.shape[2], W=grad_output.shape[3];
        const int HW=H*W, m=N*HW;
        Tensor d_input(grad_output.shape);

        const float* go = grad_output.data.data();
        const float* xn = x_norm.data.data();
        float*       di = d_input.data.data();
        float*       dg = gamma.grad.data();
        float*       db = beta.grad.data();
        const float* bv = batch_var.data.data();

        #pragma omp parallel for schedule(static)
        for (int c=0; c<C; ++c) {
            const float inv_std=1.0f/std::sqrt(bv[c]+eps);
            const float gc=gamma.data[c];
            float sum_d=0.0f, sum_dx=0.0f, dgc=0.0f, dbc=0.0f;

            for (int b=0; b<N; ++b) {
                std::span<const float> gov(go + b*(C*HW) + c*HW, HW);
                std::span<const float> xnv(xn + b*(C*HW) + c*HW, HW);
                #pragma omp simd reduction(+:sum_d,sum_dx,dgc,dbc)
                for (int hw=0; hw<HW; ++hw) {
                    sum_d  += gov[hw];
                    sum_dx += gov[hw]*xnv[hw];
                    dgc    += gov[hw]*xnv[hw];
                    dbc    += gov[hw];
                }
            }
            dg[c] += dgc;
            db[c] += dbc;

            for (int b=0; b<N; ++b) {
                std::span<const float> gov(go + b*(C*HW) + c*HW, HW);
                std::span<const float> xnv(xn + b*(C*HW) + c*HW, HW);
                std::span<float>       div(di + b*(C*HW) + c*HW, HW);
                #pragma omp simd
                for (int hw=0; hw<HW; ++hw)
                    div[hw] = (gc*inv_std/m)*(m*gov[hw] - sum_d - xnv[hw]*sum_dx);
            }
        }
        return d_input;
    }
};

} // namespace MetalNet
