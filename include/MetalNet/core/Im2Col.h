#pragma once
#include "Tensor.h"
#include <vector>

namespace MetalNet {

inline void im2col_batch(const Tensor& input, Tensor& col, int b, int kernel_size, int stride=1, int padding=0) {
    const int C  = input.shape[1], H  = input.shape[2], W  = input.shape[3];
    const int OH = (H + 2*padding - kernel_size) / stride + 1;
    const int OW = (W + 2*padding - kernel_size) / stride + 1;
    const int cols = OH * OW;

    const float* src = input.data.data() + b * (C * H * W);
    float*       dst = col.data.data();

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; ++c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int row = c * kernel_size*kernel_size + ky*kernel_size + kx;
                for (int y = 0; y < OH; ++y) {
                    for (int x = 0; x < OW; ++x) {
                        int iy = y*stride - padding + ky;
                        int ix = x*stride - padding + kx;
                        int col_idx = y*OW + x;
                        dst[row*cols + col_idx] =
                            (iy>=0 && iy<H && ix>=0 && ix<W)
                            ? src[c*(H*W) + iy*W + ix]
                            : 0.0f;
                    }
                }
            }
        }
    }
}

inline void col2im_batch(const Tensor& col, Tensor& d_in, int b, const std::vector<int>& in_shape,
                     int kernel_size, int stride=1, int padding=0) {
    const int C  = in_shape[1], H  = in_shape[2], W  = in_shape[3];
    const int OH = (H + 2*padding - kernel_size) / stride + 1;
    const int OW = (W + 2*padding - kernel_size) / stride + 1;
    const int cols = OH * OW;

    const float* src = col.data.data();
    float*       dst = d_in.data.data() + b * (C * H * W);

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; ++c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int row = c*kernel_size*kernel_size + ky*kernel_size + kx;
                for (int y = 0; y < OH; ++y) {
                    for (int x = 0; x < OW; ++x) {
                        int iy = y*stride - padding + ky;
                        int ix = x*stride - padding + kx;
                        int col_idx = y*OW + x;
                        if (iy>=0 && iy<H && ix>=0 && ix<W)
                            #pragma omp atomic
                            dst[c*(H*W) + iy*W + ix] +=
                                src[row*cols + col_idx];
                    }
                }
            }
        }
    }
}

} // namespace MetalNet
