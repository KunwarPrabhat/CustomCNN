#pragma once
#include "Layer.h"
#include "../core/Im2Col.h"
#include <cmath>
#include <algorithm>

namespace MetalNet {

class FusedConvBNReLU : public Layer {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor conv_weights, conv_biases;
    Tensor gamma, beta;
    Tensor running_mean, running_var;
    
    Tensor col_buffer, dcol_buffer;
    Tensor conv_buffer; // intermediate for training
    Tensor x_norm, batch_var, batch_mean;
    
    float momentum, eps;

    inline FusedConvBNReLU(int in_c, int out_c, int k, int s=1, int p=0, float mom=0.1f, float e=1e-5f)
        : in_channels(in_c), out_channels(out_c), kernel_size(k), stride(s), padding(p),
          momentum(mom), eps(e) {
        conv_weights = Tensor(std::vector<int>{out_c, in_c, k, k});
        conv_biases  = Tensor(out_c, 1);
        gamma = Tensor(out_c, 1); beta = Tensor(out_c, 1);
        running_mean = Tensor(out_c, 1); running_var = Tensor(out_c, 1);
        
        conv_weights.randomize();
        conv_biases.fill(0.0f); // BN biases usually fold
        gamma.fill(1.0f); beta.fill(0.0f);
        running_mean.fill(0.0f); running_var.fill(1.0f);
    }

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        const int N  = input_shapes[0][0], H = input_shapes[0][2], W = input_shapes[0][3];
        const int OH = (H + 2*padding - kernel_size) / stride + 1;
        const int OW = (W + 2*padding - kernel_size) / stride + 1;
        const int rows = in_channels * kernel_size * kernel_size;
        const int cols = OH * OW;

        output_buffer = Tensor(N, out_channels, OH, OW);
        grad_input_buffer = Tensor(input_shapes[0]);
        col_buffer = Tensor(rows, cols);
        dcol_buffer = Tensor(rows, cols);
        conv_buffer = Tensor(N, out_channels, OH, OW);
        x_norm = Tensor(N, out_channels, OH, OW);
        batch_var = Tensor(out_channels, 1);
        batch_mean = Tensor(out_channels, 1);
    }

    inline std::vector<Tensor*> get_parameters() override { return {&conv_weights, &conv_biases, &gamma, &beta}; }
    inline std::vector<Tensor*> get_states()     override { return {&conv_weights, &conv_biases, &gamma, &beta, &running_mean, &running_var}; }
    inline void zero_grad() {
        conv_weights.zero_grad(); conv_biases.zero_grad();
        gamma.zero_grad(); beta.zero_grad();
    }
    inline void update_weights(float lr) override {
        float* W=conv_weights.data.data(); float* dW=conv_weights.grad.data();
        float* b=conv_biases.data.data();  float* db=conv_biases.grad.data();
        float* g=gamma.data.data();        float* dg=gamma.grad.data();
        float* bt=beta.data.data();        float* dbt=beta.grad.data();
        #pragma omp simd
        for (int i=0;i<conv_weights.size();++i) W[i]-=lr*dW[i];
        #pragma omp simd
        for (int i=0;i<conv_biases.size();++i)  b[i]-=lr*db[i];
        #pragma omp simd
        for (int i=0;i<gamma.size();++i) g[i]-=lr*dg[i];
        #pragma omp simd
        for (int i=0;i<beta.size();++i)  bt[i]-=lr*dbt[i];
        zero_grad();
    }

    inline void forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const int N = input.shape[0];
        const int M = out_channels;
        const int K = in_channels * kernel_size * kernel_size;
        const int N_cols = output_buffer.shape[2] * output_buffer.shape[3]; 

        const float* W = conv_weights.data.data();
        const float* cb = conv_biases.data.data();
        float* out = output_buffer.data.data();
        float* cbuf = conv_buffer.data.data();
        
        constexpr int BLOCK_SIZE = 64;

        if (!is_training) {
            // Folded Inference: apply BN stats to W and b directly in registers
            const float* rm = running_mean.data.data();
            const float* rv = running_var.data.data();
            const float* gam = gamma.data.data();
            const float* bet = beta.data.data();

            for (int n = 0; n < N; ++n) {
                im2col_batch(input, col_buffer, n, kernel_size, stride, padding);
                const float* col = col_buffer.data.data();
                float* out_n = out + n * (M * N_cols);

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < M; ++i) {
                    float inv_std = gam[i] / std::sqrt(rv[i] + eps);
                    float bias_val = (cb[i] - rm[i]) * inv_std + bet[i];
                    float* o_row = out_n + i * N_cols;
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
                                float inv_std = gam[i] / std::sqrt(rv[i] + eps);
                                float* o_row = out_n + i * N_cols;
                                const float* w_row = W + i * K;
                                for (int k = k0; k < k_max; ++k) {
                                    float val = w_row[k] * inv_std; // Folded weight
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
                
                // ReLU
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < M * N_cols; ++i) {
                    if (out_n[i] < 0.0f) out_n[i] = 0.0f;
                }
            }
            return;
        }

        // Training Process
        // 1. Convolution to conv_buffer
        for (int n = 0; n < N; ++n) {
            im2col_batch(input, col_buffer, n, kernel_size, stride, padding);
            const float* col = col_buffer.data.data();
            float* cbuf_n = cbuf + n * (M * N_cols);

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < M; ++i) {
                float bias_val = cb[i];
                float* o_row = cbuf_n + i * N_cols;
                #pragma omp simd
                for (int j = 0; j < N_cols; ++j) o_row[j] = bias_val;
            }

            #pragma omp parallel for collapse(2) schedule(static)
            for (int i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
                for (int j0 = 0; j0 < N_cols; j0 += BLOCK_SIZE) {
                    int i_max = std::min(i0 + BLOCK_SIZE, M);
                    int j_max = std::min(j0 + BLOCK_SIZE, N_cols);
                    for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                        int k_max = std::min(k0 + BLOCK_SIZE, K);
                        
                        for (int i = i0; i < i_max; ++i) {
                            float* o_row = cbuf_n + i * N_cols;
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

        // 2. BatchNorm statistics & ReLU
        float* mean = batch_mean.data.data();
        float* var = batch_var.data.data();
        const int m_elements = N * N_cols;
        
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < M; ++c) {
            float acc = 0.0f;
            for (int n = 0; n < N; ++n) {
                const float* ptr = cbuf + n * (M * N_cols) + c * N_cols;
                #pragma omp simd reduction(+:acc)
                for (int j = 0; j < N_cols; ++j) acc += ptr[j];
            }
            mean[c] = acc / m_elements;
        }

        #pragma omp parallel for schedule(static)
        for (int c = 0; c < M; ++c) {
            float acc = 0.0f, mu = mean[c];
            for (int n = 0; n < N; ++n) {
                const float* ptr = cbuf + n * (M * N_cols) + c * N_cols;
                #pragma omp simd reduction(+:acc)
                for (int j = 0; j < N_cols; ++j) {
                    float d = ptr[j] - mu;
                    acc += d * d;
                }
            }
            var[c] = acc / m_elements;
        }

        for (int c = 0; c < M; ++c) {
            running_mean.data[c] = (1-momentum)*running_mean.data[c] + momentum*mean[c];
            running_var.data[c]  = (1-momentum)*running_var.data[c]  + momentum*var[c];
        }

        float* xn = x_norm.data.data();
        const float* gam = gamma.data.data();
        const float* bet = beta.data.data();

        #pragma omp parallel for schedule(static)
        for (int c = 0; c < M; ++c) {
            float inv_std = 1.0f / std::sqrt(var[c] + eps);
            float gc = gam[c], bc = bet[c], mu = mean[c];
            for (int n = 0; n < N; ++n) {
                const float* s = cbuf + n * (M * N_cols) + c * N_cols;
                float* x_ptr = xn + n * (M * N_cols) + c * N_cols;
                float* d = out + n * (M * N_cols) + c * N_cols;
                #pragma omp simd
                for (int j = 0; j < N_cols; ++j) {
                    x_ptr[j] = (s[j] - mu) * inv_std;
                    float val = gc * x_ptr[j] + bc;
                    d[j] = val > 0.0f ? val : 0.0f; // ReLU
                }
            }
        }
    }

    inline void backward(const Tensor& grad_output) override {
        const int N = grad_output.shape[0];
        const int M = out_channels;
        const int K = in_channels * kernel_size * kernel_size;
        const int N_cols = grad_output.shape[2] * grad_output.shape[3]; 

        const float* go = grad_output.data.data();
        const float* xn = x_norm.data.data();
        const float* out = output_buffer.data.data();
        
        // 1. Backprop through ReLU
        Tensor d_relu(grad_output.shape);
        float* d_relu_ptr = d_relu.data.data();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < grad_output.size(); ++i) {
            d_relu_ptr[i] = out[i] > 0.0f ? go[i] : 0.0f;
        }

        // 2. Backprop through BN
        Tensor d_bn(grad_output.shape);
        float* d_bn_ptr = d_bn.data.data();
        float* dg = gamma.grad.data();
        float* db = beta.grad.data();
        const float* var_ptr = batch_var.data.data();
        const int m_elements = N * N_cols;

        #pragma omp parallel for schedule(static)
        for (int c = 0; c < M; ++c) {
            float inv_std = 1.0f / std::sqrt(var_ptr[c] + eps);
            float gc = gamma.data[c];
            float sum_d = 0.0f, sum_dx = 0.0f;

            for (int n = 0; n < N; ++n) {
                const float* gov = d_relu_ptr + n * (M * N_cols) + c * N_cols;
                const float* xnv = xn + n * (M * N_cols) + c * N_cols;
                
                #pragma omp simd reduction(+:sum_d,sum_dx)
                for (int j = 0; j < N_cols; ++j) {
                    sum_d  += gov[j];
                    sum_dx += gov[j] * xnv[j];
                }
            }
            dg[c] += sum_dx;
            db[c] += sum_d;

            for (int n = 0; n < N; ++n) {
                const float* gov = d_relu_ptr + n * (M * N_cols) + c * N_cols;
                const float* xnv = xn + n * (M * N_cols) + c * N_cols;
                float* div       = d_bn_ptr + n * (M * N_cols) + c * N_cols;
                
                #pragma omp simd
                for (int j = 0; j < N_cols; ++j) {
                    div[j] = (gc * inv_std / m_elements) * (m_elements * gov[j] - sum_d - xnv[j] * sum_dx);
                }
            }
        }

        // 3. Backprop through Conv
        const float* d_conv = d_bn_ptr;
        const float* W = conv_weights.data.data();
        float* dW = conv_weights.grad.data();
        float* cb_grad = conv_biases.grad.data();
        
        grad_input_buffer.fill(0.0f);
        constexpr int BLOCK_SIZE = 64;

        for (int n = 0; n < N; ++n) {
            im2col_batch(*cached_input_ptr, col_buffer, n, kernel_size, stride, padding);
            const float* col = col_buffer.data.data();
            const float* go_n = d_conv + n * (M * N_cols);
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

            // d_conv_biases += sum(go_n)
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < M; ++i) {
                float acc = 0.0f;
                const float* go_row = go_n + i * N_cols;
                #pragma omp simd reduction(+:acc)
                for (int j = 0; j < N_cols; ++j) {
                    acc += go_row[j];
                }
                cb_grad[i] += acc;
            }
        }
    }
};

} // namespace MetalNet
