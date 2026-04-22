#pragma once
#include "Optimizer.h"
#include <unordered_map>
#include <cmath>

namespace MetalNet {

class Adam : public Optimizer {
public:
    float lr, b1, b2, eps;
    int   t = 0;
    std::unordered_map<Tensor*, Tensor> m, v;

    inline Adam(float l=0.001f, float b1_=0.9f, float b2_=0.999f, float e=1e-8f)
        : lr(l), b1(b1_), b2(b2_), eps(e) {}

    inline void step(std::vector<std::shared_ptr<Layer>>& layers) override {
        ++t;
        const float bc1 = 1.0f - std::pow(b1, t);
        const float bc2 = 1.0f - std::pow(b2, t);
        for (auto& layer : layers) {
            for (Tensor* p : layer->get_parameters()) {
                if (m.find(p) == m.end()) {
                    m[p]=Tensor(p->shape); m[p].fill(0.0f);
                    v[p]=Tensor(p->shape); v[p].fill(0.0f);
                }
                float* pd=p->data.data(); float* gd=p->grad.data();
                float* md=m[p].data.data(); float* vd=v[p].data.data();
                #pragma omp simd
                for (int i=0;i<p->size();++i) {
                    md[i] = b1*md[i] + (1-b1)*gd[i];
                    vd[i] = b2*vd[i] + (1-b2)*gd[i]*gd[i];
                    float mh = md[i]/bc1, vh = vd[i]/bc2;
                    pd[i] -= lr * mh / (std::sqrt(vh) + eps);
                }
                p->zero_grad();
            }
        }
    }
};

} // namespace MetalNet
