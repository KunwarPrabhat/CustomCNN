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
        weights.randomize();
        biases.fill(0.1f);
    }

    inline Tensor forward(const Tensor& input) override {
        cached_input = input;
        const int N = input.shape[0];
        Tensor output(N, output_size);

        const float* inp = input.data.data();
        const float* W   = weights.data.data();
        const float* b   = biases.data.data();
        float*       out = output.data.data();

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            std::span<const float> x_row(inp + n*input_size, input_size);
            std::span<float>       o_row(out + n*output_size, output_size);
            for (int j = 0; j < output_size; ++j) {
                float acc = b[j];
                std::span<const float> w_col(W + j, 1); // stride trick not needed; manual below
                const float* w_ptr = W + j; // weights[k][j] stored col-major? No, row-major: W[k*out+j]
                // weights shape: (in_size, out_size) → W[k*output_size + j]
                const float* wj = W; // base
                #pragma omp simd reduction(+:acc)
                for (int k = 0; k < input_size; ++k)
                    acc += x_row[k] * wj[k*output_size + j];
                o_row[j] = acc;
            }
        }
        return output;
    }

    inline Tensor backward(const Tensor& grad_output) override {
        const int N = grad_output.shape[0];
        Tensor d_input(N, input_size);

        const float* inp = cached_input.data.data();
        const float* go  = grad_output.data.data();
        const float* W   = weights.data.data();
        float*       di  = d_input.data.data();
        float*       dW  = weights.grad.data();
        float*       db  = biases.grad.data();

        // d_input = grad_output @ W^T
        #pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n) {
            std::span<const float> go_row(go  + n*output_size, output_size);
            std::span<float>       di_row(di  + n*input_size,  input_size);
            for (int k = 0; k < input_size; ++k) {
                float acc = 0.0f;
                #pragma omp simd reduction(+:acc)
                for (int j = 0; j < output_size; ++j)
                    acc += W[k*output_size + j] * go_row[j];
                di_row[k] = acc;
            }
        }

        // dW and db
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < input_size; ++k) {
            for (int j = 0; j < output_size; ++j) {
                float acc = 0.0f;
                #pragma omp simd reduction(+:acc)
                for (int n = 0; n < N; ++n)
                    acc += inp[n*input_size + k] * go[n*output_size + j];
                dW[k*output_size + j] += acc;
            }
        }
        for (int j = 0; j < output_size; ++j) {
            float acc = 0.0f;
            #pragma omp simd reduction(+:acc)
            for (int n = 0; n < N; ++n) acc += go[n*output_size + j];
            db[j] += acc;
        }
        return d_input;
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
};

} // namespace MetalNet
