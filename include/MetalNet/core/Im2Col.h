#pragma once
#include "Tensor.h"
#include <vector>

namespace MetalNet {

inline Tensor im2col(const Tensor& input, int kernel_size, int stride=1, int padding=0) {
    const int N  = input.shape[0], C  = input.shape[1];
    const int H  = input.shape[2], W  = input.shape[3];
    const int OH = (H + 2*padding - kernel_size) / stride + 1;
    const int OW = (W + 2*padding - kernel_size) / stride + 1;
    const int rows = C * kernel_size * kernel_size;
    const int cols = N * OH * OW;

    Tensor col(rows, cols);
    const float* src = input.data.data();
    float*       dst = col.data.data();

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; ++c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int row = c * kernel_size*kernel_size + ky*kernel_size + kx;
                for (int b = 0; b < N; ++b) {
                    for (int y = 0; y < OH; ++y) {
                        for (int x = 0; x < OW; ++x) {
                            int iy = y*stride - padding + ky;
                            int ix = x*stride - padding + kx;
                            int col_idx = b*(OH*OW) + y*OW + x;
                            dst[row*cols + col_idx] =
                                (iy>=0 && iy<H && ix>=0 && ix<W)
                                ? src[b*(C*H*W) + c*(H*W) + iy*W + ix]
                                : 0.0f;
                        }
                    }
                }
            }
        }
    }
    return col;
}

inline Tensor col2im(const Tensor& col, const std::vector<int>& in_shape,
                     int kernel_size, int stride=1, int padding=0) {
    const int N  = in_shape[0], C  = in_shape[1];
    const int H  = in_shape[2], W  = in_shape[3];
    const int OH = (H + 2*padding - kernel_size) / stride + 1;
    const int OW = (W + 2*padding - kernel_size) / stride + 1;
    const int cols = N * OH * OW;

    Tensor d_in(in_shape);
    const float* src = col.data.data();
    float*       dst = d_in.data.data();

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < C; ++c) {
        for (int ky = 0; ky < kernel_size; ++ky) {
            for (int kx = 0; kx < kernel_size; ++kx) {
                int row = c*kernel_size*kernel_size + ky*kernel_size + kx;
                for (int b = 0; b < N; ++b) {
                    for (int y = 0; y < OH; ++y) {
                        for (int x = 0; x < OW; ++x) {
                            int iy = y*stride - padding + ky;
                            int ix = x*stride - padding + kx;
                            int col_idx = b*(OH*OW) + y*OW + x;
                            if (iy>=0 && iy<H && ix>=0 && ix<W)
                                #pragma omp atomic
                                dst[b*(C*H*W) + c*(H*W) + iy*W + ix] +=
                                    src[row*cols + col_idx];
                        }
                    }
                }
            }
        }
    }
    return d_in;
}

} // namespace MetalNet
