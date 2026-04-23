#pragma once
#include "Layer.h"
#include <span>

namespace MetalNet {

class Dense : public Layer {
public:
    int    input_size, output_size;
    Tensor weights, biases;

    inline Dense(int in_sz, int out_sz) : input_size(in_sz), output_size(out_sz) {
        weights = Tensor(in_sz, out_sz);
        biases  = Tensor(1, out_sz);
        int fan_in = in_sz;
        weights.fill_he_init(fan_in);
        biases.fill(0.0f);
    }

    inline void compile(const std::vector<std::vector<int>>& input_shapes) override {
        if (input_shapes.empty()) throw std::runtime_error("Dense: input_shapes is empty");
        int in_features = input_shapes[0].back(); // Assuming flat or (N, D)
        if (in_features != input_size) {
            throw std::runtime_error("Dense: dimension mismatch. Expected " + std::to_string(input_size) + 
                                     ", got " + std::to_string(in_features));
        }
        const int N = input_shapes[0][0];
        output_buffer = Tensor(N, output_size);
        grad_input_buffer = Tensor(input_shapes[0]);
    }

    inline Tensor& forward(const Tensor& input) override {
        cached_input_ptr = &input;
        const int N = input.shape[0];
        const int K = input_size;
        const int M = output_size;
        
        const float* inp = input.data.data();
        const float* W   = weights.data.data();
        const float* b   = biases.data.data();
        float*       out = output_buffer.data.data();
        
        constexpr int BLOCK_SIZE = 64;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            float* o_row = out + i * M;
            #pragma omp simd
            for (int j = 0; j < M; ++j) {
                o_row[j] = b[j];
            }
        }

        #pragma omp parallel for collapse(2) schedule(static)
        for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
            for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                int i_max = std::min(i0 + BLOCK_SIZE, N);
                int j_max = std::min(j0 + BLOCK_SIZE, M);
                
                for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                    int k_max = std::min(k0 + BLOCK_SIZE, K);
                    
                    for (int i = i0; i < i_max; ++i) {
                        float* o_row = out + i * M;
                        const float* i_row = inp + i * K;
                        for (int k = k0; k < k_max; ++k) {
                            float val = i_row[k];
                            const float* w_row = W + k * M;
                            #pragma omp simd
                            for (int j = j0; j < j_max; ++j) {
                                o_row[j] += val * w_row[j];
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

        // di = go @ W^T
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
            for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
                int i_max = std::min(i0 + BLOCK_SIZE, N);
                int k_max = std::min(k0 + BLOCK_SIZE, K);
                for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                    int j_max = std::min(j0 + BLOCK_SIZE, M);
                    for (int i = i0; i < i_max; ++i) {
                        float* di_row = di + i * K;
                        const float* go_row = go + i * M;
                        for (int k = k0; k < k_max; ++k) {
                            const float* w_row = W + k * M;
                            float acc = 0.0f;
                            #pragma omp simd reduction(+:acc)
                            for (int j = j0; j < j_max; ++j) {
                                acc += go_row[j] * w_row[j];
                            }
                            di_row[k] += acc;
                        }
                    }
                }
            }
        }

        // dW = inp^T @ go
        #pragma omp parallel for collapse(2) schedule(static)
        for (int k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
            for (int j0 = 0; j0 < M; j0 += BLOCK_SIZE) {
                int k_max = std::min(k0 + BLOCK_SIZE, K);
                int j_max = std::min(j0 + BLOCK_SIZE, M);
                for (int i0 = 0; i0 < N; i0 += BLOCK_SIZE) {
                    int i_max = std::min(i0 + BLOCK_SIZE, N);
                    for (int k = k0; k < k_max; ++k) {
                        float* dw_row = dW + k * M;
                        for (int i = i0; i < i_max; ++i) {
                            float val = inp[i * K + k]; // inp^T
                            const float* go_row = go + i * M;
                            #pragma omp simd
                            for (int j = j0; j < j_max; ++j) {
                                dw_row[j] += val * go_row[j];
                            }
                        }
                    }
                }
            }
        }

        // db = sum(go)
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < M; ++j) {
            float acc = 0.0f;
            #pragma omp simd reduction(+:acc)
            for (int i = 0; i < N; ++i) {
                acc += go[i * M + j];
            }
            db[j] += acc;
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
