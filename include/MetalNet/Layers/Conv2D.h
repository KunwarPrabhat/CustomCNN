#pragma once
#include "Layer.h"
#include <omp.h>
#include <immintrin.h>
#include "../core/Im2Col.h"

namespace MetalNet {

class Conv2D : public Layer {
public:
    int in_channels, out_channels, kernel_size, stride, padding;
    Tensor weights, biases;
    std::vector<Tensor> col_buffers, dcol_buffers;
    std::vector<Tensor> local_grad_weights, local_grad_biases;

    inline Conv2D(int in_c, int out_c, int k, int s=1, int p=0)
        : in_channels(in_c), out_channels(out_c), kernel_size(k), stride(s), padding(p) {
        // [kernel_size * kernel_size * in_channels, out_channels]
        weights = Tensor(std::vector<int>{k, k, in_c, out_c}); 
        biases  = Tensor(out_c, 1);
        int fan_in = in_c * k * k;
        weights.fill_he_init(fan_in);
        biases.fill(0.0f);
    }

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        int N = input_shapes[0][0], H, W, C;
        // Auto-detect format based on in_channels position (useful if main.cpp seeds via NCHW)
        if (input_shapes[0].size() == 4 && input_shapes[0][1] == in_channels && input_shapes[0][3] != in_channels) {
            C = input_shapes[0][1]; H = input_shapes[0][2]; W = input_shapes[0][3];
        } else if (input_shapes[0].size() == 4) {
            H = input_shapes[0][1]; W = input_shapes[0][2]; C = input_shapes[0][3];
        } else {
             throw std::runtime_error("Conv2D: Invalid dimensions");
        }
        if (C != in_channels) throw std::runtime_error("Conv2D: channel mismatch.");

        const int OH = (H + 2*padding - kernel_size) / stride + 1;
        const int OW = (W + 2*padding - kernel_size) / stride + 1;
        const int rows = OH * OW;
        const int cols = in_channels * kernel_size * kernel_size;

        output_buffer = Tensor(N, OH, OW, out_channels);
        grad_input_buffer = Tensor(input_shapes[0]);

        col_buffers.clear(); dcol_buffers.clear();
        for (int i = 0; i < N; ++i) {
            col_buffers.emplace_back(rows, cols);     // (N_cols, K)
            dcol_buffers.emplace_back(rows, cols);
        }
        int max_threads = omp_get_max_threads();
        local_grad_weights.clear(); local_grad_biases.clear();
        for (int t = 0; t < max_threads; ++t) {
            local_grad_weights.emplace_back(std::vector<int>{kernel_size, kernel_size, in_channels, out_channels});
            local_grad_biases.emplace_back(out_channels, 1);
        }
    }

    inline Tensor& forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const int N = input.shape[0];
        const int N_cols = output_buffer.shape[1] * output_buffer.shape[2]; 
        const int M = out_channels;
        const int K_dim = in_channels * kernel_size * kernel_size;

        const float* W = weights.data.data();
        float* out = output_buffer.data.data();
        const float* b = biases.data.data();
        constexpr int BLOCK_SIZE = 64;

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            im2col_batch(input, col_buffers[n], n, kernel_size, stride, padding);
            const float* col = col_buffers[n].data.data();
            float* out_n = out + n * (N_cols * M);

            for (int i0 = 0; i0 < N_cols; i0 += BLOCK_SIZE) {
                int i_max = std::min(i0 + BLOCK_SIZE, N_cols);
                
                for (int i = i0; i < i_max; ++i) {
                    float* o_row = out_n + i * M;
                    #ifdef __AVX2__
                    int j = 0;
                    for (; j + 7 < M; j += 8) {
                        _mm256_storeu_ps(o_row + j, _mm256_loadu_ps(b + j));
                    }
                    for (; j < M; ++j) o_row[j] = b[j];
                    #else
                    #pragma omp simd
                    for (int j = 0; j < M; ++j) o_row[j] = b[j];
                    #endif
                }

                for (int k0 = 0; k0 < K_dim; k0 += BLOCK_SIZE) {
                    int k_max = std::min(k0 + BLOCK_SIZE, K_dim);
                    for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                        int j_max = std::min(j0 + BLOCK_SIZE, M);
                        for (int i = i0; i < i_max; ++i) {
                            float* o_row = out_n + i * M;
                            const float* col_row = col + i * K_dim;
                            for (int k = k0; k < k_max; ++k) {
                                float val = col_row[k];
                                const float* w_row = W + k * M;
                                #ifdef __AVX2__
                                __m256 v_val = _mm256_set1_ps(val);
                                int j = j0;
                                for (; j + 15 < j_max; j += 16) {
                                    __m256 out_v0 = _mm256_loadu_ps(o_row + j);
                                    __m256 out_v1 = _mm256_loadu_ps(o_row + j + 8);
                                    __m256 w_v0 = _mm256_loadu_ps(w_row + j);
                                    __m256 w_v1 = _mm256_loadu_ps(w_row + j + 8);
                                    out_v0 = _mm256_fmadd_ps(v_val, w_v0, out_v0);
                                    out_v1 = _mm256_fmadd_ps(v_val, w_v1, out_v1);
                                    _mm256_storeu_ps(o_row + j, out_v0);
                                    _mm256_storeu_ps(o_row + j + 8, out_v1);
                                }
                                for (; j + 7 < j_max; j += 8) {
                                    __m256 out_v = _mm256_loadu_ps(o_row + j);
                                    __m256 w_v = _mm256_loadu_ps(w_row + j);
                                    out_v = _mm256_fmadd_ps(v_val, w_v, out_v);
                                    _mm256_storeu_ps(o_row + j, out_v);
                                }
                                for (; j < j_max; ++j) {
                                    o_row[j] += val * w_row[j];
                                }
                                #else
                                #pragma omp simd
                                for (int j = j0; j < j_max; ++j) {
                                    o_row[j] += val * w_row[j];
                                }
                                #endif
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
        const int N_cols = grad_output.shape[1] * grad_output.shape[2]; 
        const int M = out_channels;
        const int K_dim = in_channels * kernel_size * kernel_size;

        const float* go = grad_output.data.data();
        const float* W  = weights.data.data();
        float* dW       = weights.grad.data();
        float* db       = biases.grad.data();
        
        grad_input_buffer.fill(0.0f);
        constexpr int BLOCK_SIZE = 64;

        int max_threads = omp_get_max_threads();
        for (int t = 0; t < max_threads; ++t) {
            local_grad_weights[t].fill(0.0f);
            local_grad_biases[t].fill(0.0f);
        }

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            int tid = omp_get_thread_num();
            float* local_dW = local_grad_weights[tid].data.data();
            float* local_db = local_grad_biases[tid].data.data();

            im2col_batch(*cached_input_ptr, col_buffers[n], n, kernel_size, stride, padding);
            const float* col = col_buffers[n].data.data();
            const float* go_n = go + n * (N_cols * M);
            float* dcol = dcol_buffers[n].data.data();
            
            dcol_buffers[n].fill(0.0f);
            
            // dcol = go_n @ W^T  [N_cols, K_dim] = [N_cols, M] @ [M, K_dim] Wait! W is [K_dim, M]. So go_n @ W^T is [N_cols, K_dim].
            for (int k0 = 0; k0 < K_dim; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K_dim);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int i = 0; i < N_cols; ++i) {
                        float* dcol_row = dcol + i * K_dim;
                        const float* go_row = go_n + i * M;
                        for (int k = k0; k < k_max; ++k) {
                            const float* w_row = W + k * M;
                            float acc = 0.0f;
                            #pragma omp simd reduction(+:acc)
                            for (int j = j0; j < j_max; ++j) {
                                acc += go_row[j] * w_row[j];
                            }
                            dcol_row[k] += acc;
                        }
                    }
                }
            }
            
            col2im_batch(dcol_buffers[n], grad_input_buffer, n, cached_input_ptr->shape, kernel_size, stride, padding);

            // local_dW += col^T @ go_n   [K_dim, M] = [K_dim, N_cols] @ [N_cols, M]
            for (int k0 = 0; k0 < K_dim; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K_dim);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int k = k0; k < k_max; ++k) {
                        float* dw_row = local_dW + k * M;
                        for (int i = 0; i < N_cols; ++i) {
                            float val = col[i * K_dim + k]; 
                            const float* go_row = go_n + i * M;
                            #ifdef __AVX2__
                            __m256 v_val = _mm256_set1_ps(val);
                            int j = j0;
                            for (; j + 7 < j_max; j += 8) {
                                __m256 h_v = _mm256_loadu_ps(dw_row + j);
                                __m256 g_v = _mm256_loadu_ps(go_row + j);
                                h_v = _mm256_fmadd_ps(v_val, g_v, h_v);
                                _mm256_storeu_ps(dw_row + j, h_v);
                            }
                            for (; j < j_max; ++j) {
                                dw_row[j] += val * go_row[j];
                            }
                            #else
                            #pragma omp simd
                            for (int j = j0; j < j_max; ++j) {
                                dw_row[j] += val * go_row[j];
                            }
                            #endif
                        }
                    }
                }
            }

            // local_db += sum(go_n, axis=0) (since go_n is [N_cols, M])
            for (int i = 0; i < N_cols; ++i) {
                const float* go_row = go_n + i * M;
                #pragma omp simd
                for (int j = 0; j < M; ++j) {
                    local_db[j] += go_row[j];
                }
            }
        }

        for (int t = 0; t < max_threads; ++t) {
            const float* l_dW = local_grad_weights[t].data.data();
            const float* l_db = local_grad_biases[t].data.data();
            #pragma omp simd
            for (int i = 0; i < K_dim * M; ++i) {
                dW[i] += l_dW[i];
            }
            #pragma omp simd
            for (int i = 0; i < M; ++i) {
                db[i] += l_db[i];
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
