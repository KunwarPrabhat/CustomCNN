#pragma once
#include "Tensor.h"
#include <vector>

namespace MetalNet {

inline void im2col_batch(const Tensor& input, Tensor& col, int b, int kernel_size, int stride=1, int padding=0) {
    const int H  = input.shape[1], W  = input.shape[2], C  = input.shape[3];
    const int OH = (H + 2*padding - kernel_size) / stride + 1;
    const int OW = (W + 2*padding - kernel_size) / stride + 1;
    const int cols = kernel_size * kernel_size * C;

    const float* src = input.data.data() + b * (H * W * C);
    float*       dst = col.data.data();

    for (int y = 0; y < OH; ++y) {
        for (int x = 0; x < OW; ++x) {
            int out_idx = y * OW + x;
            float* dst_row = dst + out_idx * cols;
            
            for (int ky = 0; ky < kernel_size; ++ky) {
                int iy = y*stride - padding + ky;
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int ix = x*stride - padding + kx;
                    int patch_idx = (ky * kernel_size + kx) * C;
                    
                    if (iy>=0 && iy<H && ix>=0 && ix<W) {
                        const float* src_row = src + (iy * W + ix) * C;
                        #pragma omp simd
                        for (int c = 0; c < C; ++c) {
                            dst_row[patch_idx + c] = src_row[c];
                        }
                    } else {
                        #pragma omp simd
                        for (int c = 0; c < C; ++c) {
                            dst_row[patch_idx + c] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

inline void col2im_batch(const Tensor& col, Tensor& d_in, int b, const std::vector<int>& in_shape,
                     int kernel_size, int stride=1, int padding=0) {
    const int H  = in_shape[1], W  = in_shape[2], C  = in_shape[3];
    const int OH = (H + 2*padding - kernel_size) / stride + 1;
    const int OW = (W + 2*padding - kernel_size) / stride + 1;
    const int cols = kernel_size * kernel_size * C;

    const float* src = col.data.data();
    float*       dst = d_in.data.data() + b * (H * W * C);

    for (int y = 0; y < OH; ++y) {
        for (int x = 0; x < OW; ++x) {
            int out_idx = y * OW + x;
            const float* src_row = src + out_idx * cols;
            
            for (int ky = 0; ky < kernel_size; ++ky) {
                int iy = y*stride - padding + ky;
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int ix = x*stride - padding + kx;
                    int patch_idx = (ky * kernel_size + kx) * C;
                    
                    if (iy>=0 && iy<H && ix>=0 && ix<W) {
                        float* dst_row = dst + (iy * W + ix) * C;
                        #pragma omp simd
                        for (int c = 0; c < C; ++c) {
                            dst_row[c] += src_row[patch_idx + c];
                        }
                    }
                }
            }
        }
    }
}

inline Tensor im2col(const Tensor& input, int kernel_size, int stride=1, int padding=0) {
    const int H  = input.shape[1], W  = input.shape[2], C = input.shape[3];
    const int OH = (H + 2*padding - kernel_size) / stride + 1;
    const int OW = (W + 2*padding - kernel_size) / stride + 1;
    Tensor col(OH * OW, kernel_size * kernel_size * C);
    im2col_batch(input, col, 0, kernel_size, stride, padding);
    return col;
}

} // namespace MetalNet
