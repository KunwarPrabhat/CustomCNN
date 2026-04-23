#pragma once
#include "Layer.h"

#include "../core/Im2Col.h"

namespace MetalNet {

class Conv2D : public Layer {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor weights, biases;
    Tensor col_buffer, dcol_buffer;

    inline Conv2D(int in_c, int out_c, int k, int s=1, int p=0)
        : in_channels(in_c), out_channels(out_c), kernel_size(k), stride(s), padding(p) {
        weights = Tensor(std::vector<int>{out_c, in_c, k, k});
        biases  = Tensor(out_c, 1);
        int fan_in = in_c * k * k;
        weights.fill_he_init(fan_in);
        biases.fill(0.0f);
    }

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        if (input_shapes.empty()) throw std::runtime_error("Conv2D: input_shapes is empty");
        if (input_shapes[0].size() != 4) throw std::runtime_error("Conv2D: input must have 4 dims (N,C,H,W)");
        if (input_shapes[0][1] != in_channels) {
            throw std::runtime_error("Conv2D: channel mismatch. Expected " + std::to_string(in_channels) + 
                                     ", got " + std::to_string(input_shapes[0][1]));
        }

        const int N  = input_shapes[0][0], H = input_shapes[0][2], W = input_shapes[0][3];
        const int OH = (H + 2*padding - kernel_size) / stride + 1;
        const int OW = (W + 2*padding - kernel_size) / stride + 1;
        const int rows = in_channels * kernel_size * kernel_size;
        const int cols = OH * OW;

        output_buffer = Tensor(N, out_channels, OH, OW);
        grad_input_buffer = Tensor(input_shapes[0]);
        col_buffer = Tensor(rows, cols);
        dcol_buffer = Tensor(rows, cols);
    }

    inline Tensor& forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const int N = input.shape[0];
        const int M = out_channels;
        const int K = in_channels * kernel_size * kernel_size;
        const int N_cols = output_buffer.shape[2] * output_buffer.shape[3]; 

        const float* W = weights.data.data();
        float* out = output_buffer.data.data();
        const float* b = biases.data.data();
        
        constexpr int BLOCK_SIZE = 64;

        for (int n = 0; n < N; ++n) {
            im2col_batch(input, col_buffer, n, kernel_size, stride, padding);
            const float* col = col_buffer.data.data();
            float* out_n = out + n * (M * N_cols);

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < M; ++i) {
                float* o_row = out_n + i * N_cols;
                float bias_val = b[i];
                #pragma omp simd
                for (int j = 0; j < N_cols; ++j) {
                    o_row[j] = bias_val;
                }
            }

            #pragma omp parallel for collapse(2) schedule(static)
            for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
                for (int j0 = 0; j0 < N_cols; j0 += BLOCK_SIZE) {
                    int i_max = std::min(i0 + BLOCK_SIZE, M);
                    int j_max = std::min(j0 + BLOCK_SIZE, N_cols);
                    for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                        int k_max = std::min(k0 + BLOCK_SIZE, K);
                        
                        for (int i = i0; i < i_max; ++i) {
                            float* o_row = out_n + i * N_cols;
                            const float* w_row = W + i * K;
                            for (int k = k0; k < k_max; ++k) {
                                float val = w_row[k];
                                const float* col_row = col + k * N_cols;
                                #pragma omp simd
                                for (int j = j0; j < j_max; ++j) {
                                    o_row[j] += val * col_row[j];
                                }
                            }
                        }
                    }
                }
            }
        }
        return output_buffer;
    }

    inline void backward(const Tensor& grad_output) override {
        const int N = grad_output.shape[0];
        const int M = out_channels;
        const int K = in_channels * kernel_size * kernel_size;
        const int N_cols = grad_output.shape[2] * grad_output.shape[3]; 

        const float* go = grad_output.data.data();
        const float* W  = weights.data.data();
        float* dW       = weights.grad.data();
        float* db       = biases.grad.data();
        
        grad_input_buffer.fill(0.0f);
        constexpr int BLOCK_SIZE = 64;

        for (int n = 0; n < N; ++n) {
            im2col_batch(*cached_input_ptr, col_buffer, n, kernel_size, stride, padding);
            const float* col = col_buffer.data.data();
            const float* go_n = go + n * (M * N_cols);
            float* dcol = dcol_buffer.data.data();
            
            dcol_buffer.fill(0.0f);
            
            // dcol = W^T @ go_n
            #pragma omp parallel for collapse(2) schedule(static)
            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                for (int j0 = 0; j0 < N_cols; j0 += BLOCK_SIZE) {
                    int k_max = std::min(k0 + BLOCK_SIZE, K);
                    int j_max = std::min(j0 + BLOCK_SIZE, N_cols);
                    for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
                        int i_max = std::min(i0 + BLOCK_SIZE, M);
                        for (int k = k0; k < k_max; ++k) {
                            float* dcol_row = dcol + k * N_cols;
                            for (int i = i0; i < i_max; ++i) {
                                float val = W[i * K + k];
                                const float* go_row = go_n + i * N_cols;
                                #pragma omp simd
                                for (int j = j0; j < j_max; ++j) {
                                    dcol_row[j] += val * go_row[j];
                                }
                            }
                        }
                    }
                }
            }
            
            col2im_batch(dcol_buffer, grad_input_buffer, n, cached_input_ptr->shape, kernel_size, stride, padding);

            // dW += go_n @ col^T
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
                for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                    int i_max = std::min(i0 + BLOCK_SIZE, M);
                    int k_max = std::min(k0 + BLOCK_SIZE, K);
                    for (int j0 = 0; j0 < N_cols; j0 += BLOCK_SIZE) {
                        int j_max = std::min(j0 + BLOCK_SIZE, N_cols);
                        for (int i = i0; i < i_max; ++i) {
                            float* dw_row = dW + i * K;
                            const float* go_row = go_n + i * N_cols;
                            for (int k = k0; k < k_max; ++k) {
                                const float* col_row = col + k * N_cols;
                                float acc = 0.0f;
                                #pragma omp simd reduction(+:acc)
                                for (int j = j0; j < j_max; ++j) {
                                    acc += go_row[j] * col_row[j];
                                }
                                dw_row[k] += acc;
                            }
                        }
                    }
                }
            }

            // db += sum(go_n, axis=1)
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < M; ++i) {
                float acc = 0.0f;
                const float* go_row = go_n + i * N_cols;
                #pragma omp simd reduction(+:acc)
                for (int j = 0; j < N_cols; ++j) {
                    acc += go_row[j];
                }
                db[i] += acc;
            }
        }
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
    inline std::string name() const override { return "Conv2D"; }
};

} // namespace MetalNet
