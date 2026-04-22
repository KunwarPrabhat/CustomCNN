#pragma once
#include "../core/Tensor.h"
#include <cmath>
#include <limits>
#include <algorithm>

namespace MetalNet {

class Loss {
public:
    virtual ~Loss() = default;
    virtual float  forward (const Tensor& p, const Tensor& t) = 0;
    virtual Tensor backward(const Tensor& p, const Tensor& t) = 0;
};

class MSELoss : public Loss {
public:
    inline float forward(const Tensor& p, const Tensor& t) override {
        float loss=0.0f; const int N=p.shape[0];
        const float* pd=p.data.data(); const float* td=t.data.data();
        #pragma omp parallel for simd reduction(+:loss)
        for (int i=0;i<p.size();++i) { float d=pd[i]-td[i]; loss+=d*d; }
        return loss/N;
    }
    inline Tensor backward(const Tensor& p, const Tensor& t) override {
        const int N=p.shape[0]; Tensor d(p.shape);
        const float* pd=p.data.data(); const float* td=t.data.data(); float* di=d.data.data();
        #pragma omp simd
        for (int i=0;i<p.size();++i) di[i]=2.0f*(pd[i]-td[i])/N;
        return d;
    }
};

class CrossEntropyLoss : public Loss {
public:
    inline float forward(const Tensor& p, const Tensor& t) override {
        const int N=p.shape[0], C=p.shape[1]; float loss=0.0f;
        #pragma omp parallel for reduction(+:loss)
        for (int i=0;i<N;++i) {
            float mx=-std::numeric_limits<float>::infinity();
            for (int j=0;j<C;++j) mx=std::max(mx,p(i,j));
            float se=0; for (int j=0;j<C;++j) se+=std::exp(p(i,j)-mx);
            for (int j=0;j<C;++j) if(t(i,j)>0)
                loss-=t(i,j)*std::log(std::exp(p(i,j)-mx)/se+1e-9f);
        }
        return loss/N;
    }
    inline Tensor backward(const Tensor& p, const Tensor& t) override {
        const int N=p.shape[0], C=p.shape[1]; Tensor d(p.shape);
        for (int i=0;i<N;++i) {
            float mx=-std::numeric_limits<float>::infinity();
            for (int j=0;j<C;++j) mx=std::max(mx,p(i,j));
            float se=0; for (int j=0;j<C;++j) se+=std::exp(p(i,j)-mx);
            for (int j=0;j<C;++j) d(i,j)=(std::exp(p(i,j)-mx)/se-t(i,j))/N;
        }
        return d;
    }
};

} // namespace MetalNet
