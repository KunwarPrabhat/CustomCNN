#pragma once
#include "Dataset.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <random>

namespace MetalNet {

class DataLoader {
public:
    Dataset*         dataset;
    int              batch_size;
    bool             shuffle;
    std::vector<int> indices;
    int              current_idx;

    inline DataLoader(Dataset* ds, int bs, bool sh=true)
        : dataset(ds), batch_size(bs), shuffle(sh), current_idx(0) {
        if (dataset && dataset->images.dims()>0) {
            int n=dataset->images.shape[0];
            indices.resize(n);
            for (int i=0;i<n;++i) indices[i]=i;
            reset();
        }
    }

    inline bool has_next() const {
        if (!dataset || dataset->images.dims()==0) return false;
        return current_idx < dataset->images.shape[0];
    }

    inline void reset() {
        current_idx=0;
        if (shuffle && !indices.empty()) {
            std::mt19937 g(42);
            std::shuffle(indices.begin(), indices.end(), g);
        }
    }

    inline std::pair<Tensor,Tensor> next_batch() {
        const int ns=dataset->images.shape[0];
        const int ei=std::min(current_idx+batch_size, ns);
        const int bs=ei-current_idx;
        const int C=dataset->images.shape[1], H=dataset->images.shape[2], W=dataset->images.shape[3];
        const int NC=dataset->labels.shape[1];
        Tensor bx(bs,C,H,W), by(bs,NC);
        const float* src_x=dataset->images.data.data();
        const float* src_y=dataset->labels.data.data();
        float* dx=bx.data.data(); float* dy=by.data.data();
        for (int b=0;b<bs;++b) {
            int idx=indices[current_idx+b];
            std::span<const float> sx(src_x+idx*(C*H*W), C*H*W);
            std::span<float>       tx(dx   +b  *(C*H*W), C*H*W);
            std::copy(sx.begin(), sx.end(), tx.begin());
            std::span<const float> sy(src_y+idx*NC, NC);
            std::span<float>       ty(dy   +b  *NC, NC);
            std::copy(sy.begin(), sy.end(), ty.begin());
        }
        current_idx=ei;
        return {bx, by};
    }
};

} // namespace MetalNet
