#include "Im2Col.h"

Tensor im2col(const Tensor& input, int kernel_size, int stride, int padding) {
    int batches = input.shape[0];
    int channels = input.shape[1];
    int height = input.shape[2];
    int width = input.shape[3];

    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;

    int cols = batches * out_h * out_w;
    int rows = channels * kernel_size * kernel_size;

    Tensor col_tensor(rows, cols);
    col_tensor.fill(0.0f);

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int row = c * (kernel_size * kernel_size) + ky * kernel_size + kx;
                    for (int y = 0; y < out_h; ++y) {
                        for (int x = 0; x < out_w; ++x) {
                            int in_y = y * stride - padding + ky;
                            int in_x = x * stride - padding + kx;
                            int col = b * (out_h * out_w) + y * out_w + x;
                            
                            if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                                col_tensor(row, col) = input(b, c, in_y, in_x);
                            }
                        }
                    }
                }
            }
        }
    }
    return col_tensor;
}

Tensor col2im(const Tensor& col_tensor, const std::vector<int>& input_shape, int kernel_size, int stride, int padding) {
    int batches = input_shape[0];
    int channels = input_shape[1];
    int height = input_shape[2];
    int width = input_shape[3];

    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;

    Tensor d_input(batches, channels, height, width);
    d_input.fill(0.0f);

    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batches; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int row = c * (kernel_size * kernel_size) + ky * kernel_size + kx;
                    for (int y = 0; y < out_h; ++y) {
                        for (int x = 0; x < out_w; ++x) {
                            int in_y = y * stride - padding + ky;
                            int in_x = x * stride - padding + kx;
                            int col = b * (out_h * out_w) + y * out_w + x;
                            
                            if (in_y >= 0 && in_y < height && in_x >= 0 && in_x < width) {
                                d_input(b, c, in_y, in_x) += col_tensor(row, col);
                            }
                        }
                    }
                }
            }
        }
    }
    return d_input;
}
