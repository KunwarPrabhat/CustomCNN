#pragma once
#include "Layer.h"
#include <span>
#include <omp.h>
#include <immintrin.h>

namespace MetalNet {

class Dense : public Layer {
public:
    int    input_size, output_size;
    Tensor weights, biases;
    std::vector<Tensor> local_grad_weights, local_grad_biases;

    inline Dense(int in_sz, int out_sz) : input_size(in_sz), output_size(out_sz) {
        weights = Tensor(in_sz, out_sz);
        biases  = Tensor(1, out_sz);
        int fan_in = in_sz;
        weights.fill_he_init(fan_in);
        biases.fill(0.0f);
    }

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        if (input_shapes.empty()) throw std::runtime_error("Dense: input_shapes is empty");
        int in_features = input_shapes[0].back(); 
        if (in_features != input_size) {
            throw std::runtime_error("Dense: dimension mismatch.");
        }
        const int N = input_shapes[0][0];
        output_buffer = Tensor(N, output_size);
        grad_input_buffer = Tensor(input_shapes[0]);

        int max_threads = omp_get_max_threads();
        local_grad_weights.clear(); local_grad_biases.clear();
        for (int t = 0; t < max_threads; ++t) {
            local_grad_weights.emplace_back(input_size, output_size);
            local_grad_biases.emplace_back(1, output_size);
        }
    }
inline Tensor& forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const int N = input.shape[0];
        const int K = input_size;
        const int M = output_size;
        
        const float* inp = input.data.data();
        const float* W   = weights.data.data();
        const float* b   = biases.data.data();
        float* out = output_buffer.data.data();
        
        constexpr int BLOCK_SIZE = 64;
        
        // [SCALING FIX] Parallelize directly over N. 
        // For Batch 16, exactly 16 threads will wake up and process 1 image each!
        #pragma omp parallel for schedule(static) if(N > 1)
        for (int i = 0; i < N; ++i) {
            float* o_row = out + i * M;
            
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

            const float* i_row = inp + i * K;
            // Keep blocking for Cache Locality, but inside the thread!
            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int k = k0; k < k_max; ++k) {
                        float val = i_row[k];
                        const float* w_row = W + k * M;
                        
                        #ifdef __AVX2__
                        __m256 v_val = _mm256_set1_ps(val);
                        int jj = j0;
                        for (; jj + 15 < j_max; jj += 16) {
                            __m256 out_v0 = _mm256_loadu_ps(o_row + jj);
                            __m256 out_v1 = _mm256_loadu_ps(o_row + jj + 8);
                            out_v0 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + jj), out_v0);
                            out_v1 = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + jj + 8), out_v1);
                            _mm256_storeu_ps(o_row + jj, out_v0);
                            _mm256_storeu_ps(o_row + jj + 8, out_v1);
                        }
                        for (; jj + 7 < j_max; jj += 8) {
                            __m256 out_v = _mm256_loadu_ps(o_row + jj);
                            out_v = _mm256_fmadd_ps(v_val, _mm256_loadu_ps(w_row + jj), out_v);
                            _mm256_storeu_ps(o_row + jj, out_v);
                        }
                        for (; jj < j_max; ++jj) o_row[jj] += val * w_row[jj];
                        #else
                        #pragma omp simd
                        for (int jj = j0; jj < j_max; ++jj) {
                            o_row[jj] += val * w_row[jj];
                        }
                        #endif
                    }
                }
            }
        }
        return output_buffer;
    }

    inline void backward(const Tensor& grad_output) override {
        const int N = grad_output.shape[0];
        const int K = input_size;
        const int M = output_size;

        const float* inp = cached_input_ptr->data.data();
        const float* go  = grad_output.data.data();
        const float* W   = weights.data.data();
        float*       di  = grad_input_buffer.data.data();
        float*       dW  = weights.grad.data();
        float*       db  = biases.grad.data();
        
        grad_input_buffer.fill(0.0f);
        constexpr int BLOCK_SIZE = 64;

        int max_threads = omp_get_max_threads();
        for (int t = 0; t < max_threads; ++t) {
            local_grad_weights[t].fill(0.0f);
            local_grad_biases[t].fill(0.0f);
        }

        #pragma omp parallel for schedule(static)
        for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
            int i_max = std::min(i0 + BLOCK_SIZE, N);
            int tid = omp_get_thread_num();
            float* local_dW = local_grad_weights[tid].data.data();
            float* local_db = local_grad_biases[tid].data.data();

            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int i = i0; i < i_max; ++i) {
                        float* di_row = di + i * K;
                        const float* go_row = go + i * M;
                        for (int k = k0; k < k_max; ++k) {
                            const float* w_row = W + k * M;
                            float acc = 0.0f;
                            #ifdef __AVX2__
                            __m256 v_acc = _mm256_setzero_ps();
                            int j = j0;
                            for (; j + 31 < j_max; j += 32) {
                                __m256 go0 = _mm256_loadu_ps(go_row + j);
                                __m256 w0 = _mm256_loadu_ps(w_row + j);
                                v_acc = _mm256_fmadd_ps(go0, w0, v_acc);
                                
                                __m256 go1 = _mm256_loadu_ps(go_row + j + 8);
                                __m256 w1 = _mm256_loadu_ps(w_row + j + 8);
                                v_acc = _mm256_fmadd_ps(go1, w1, v_acc);
                                
                                __m256 go2 = _mm256_loadu_ps(go_row + j + 16);
                                __m256 w2 = _mm256_loadu_ps(w_row + j + 16);
                                v_acc = _mm256_fmadd_ps(go2, w2, v_acc);
                                
                                __m256 go3 = _mm256_loadu_ps(go_row + j + 24);
                                __m256 w3 = _mm256_loadu_ps(w_row + j + 24);
                                v_acc = _mm256_fmadd_ps(go3, w3, v_acc);
                            }
                            for (; j + 7 < j_max; j += 8) {
                                __m256 go0 = _mm256_loadu_ps(go_row + j);
                                __m256 w0 = _mm256_loadu_ps(w_row + j);
                                v_acc = _mm256_fmadd_ps(go0, w0, v_acc);
                            }
                            alignas(32) float acc_arr[8];
                            _mm256_store_ps(acc_arr, v_acc);
                            for (int a = 0; a < 8; ++a) acc += acc_arr[a];
                            for (; j < j_max; ++j) {
                                acc += go_row[j] * w_row[j];
                            }
                            #else
                            #pragma omp simd reduction(+:acc)
                            for (int j = j0; j < j_max; ++j) {
                                acc += go_row[j] * w_row[j];
                            }
                            #endif
                            di_row[k] += acc;
                        }
                    }
                }
            }

            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int k = k0; k < k_max; ++k) {
                        float* dw_row = local_dW + k * M;
                        for (int i = i0; i < i_max; ++i) {
                            float val = inp[i * K + k]; // inp^T
                            const float* go_row = go + i * M;
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

            for (int i = i0; i < i_max; ++i) {
                const float* go_row = go + i * M;
                #ifdef __AVX2__
                int j = 0;
                for (; j + 7 < M; j += 8) {
                    __m256 db_v = _mm256_loadu_ps(local_db + j);
                    __m256 go_v = _mm256_loadu_ps(go_row + j);
                    db_v = _mm256_add_ps(db_v, go_v);
                    _mm256_storeu_ps(local_db + j, db_v);
                }
                for (; j < M; ++j) {
                    local_db[j] += go_row[j];
                }
                #else
                #pragma omp simd
                for (int j = 0; j < M; ++j) {
                    local_db[j] += go_row[j];
                }
                #endif
            }
        }

        for (int t = 0; t < max_threads; ++t) {
            const float* l_dW = local_grad_weights[t].data.data();
            const float* l_db = local_grad_biases[t].data.data();
            #pragma omp simd
            for (int i = 0; i < K * M; ++i) dW[i] += l_dW[i];
            #pragma omp simd
            for (int i = 0; i < M; ++i) db[i] += l_db[i];
        }
    }

    inline void update_weights(float lr) override {
        float* W  = weights.data.data(); float* dW = weights.grad.data();
        float* b  = biases.data.data();  float* db = biases.grad.data();
        #pragma omp simd
        for (int i = 0; i < weights.size(); ++i) W[i] -= lr * dW[i];
        #pragma omp simd
        for (int i = 0; i < biases.size();  ++i) b[i] -= lr * db[i];
        weights.zero_grad(); biases.zero_grad();
    }

    inline std::vector<Tensor*> get_parameters() override { return {&weights, &biases}; }
    inline std::string name() const override { return "Dense"; }
};

} // namespace MetalNet
