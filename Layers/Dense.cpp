#include "Dense.h"

Dense::Dense(int in_size, int out_size) : input_size(in_size), output_size(out_size) {
    weights = Tensor(in_size, out_size);
    biases = Tensor(1, out_size);

    weights.randomize();
    biases.fill(0.1f);
}

Tensor Dense::forward(const Tensor& input) {
    cached_input = input;

    int batches = input.shape[0];
    Tensor output(batches, output_size);
    output.fill(0.0f);

    #pragma omp parallel for
    for (int b = 0; b < batches; ++b) {
        for (int j = 0; j < output_size; ++j) {
            float sum = biases(0, j);
            for (int k = 0; k < input_size; ++k) {
                sum += input(b, k) * weights(k, j);
            }
            output(b, j) = sum;
        }
    }

    return output;
}

Tensor Dense::backward(const Tensor& grad_output) {
    int batches = grad_output.shape[0];
    Tensor d_input(batches, input_size);
    d_input.fill(0.0f);

    // 1. Compute d_input safely over batches
    #pragma omp parallel for
    for (int b = 0; b < batches; ++b) {
        for (int k = 0; k < input_size; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < output_size; ++j) {
                sum += weights(k, j) * grad_output(b, j);
            }
            d_input(b, k) = sum;
        }
    }

    // 2. Compute parameter gradients safely over output features (avoids atomic locks)
    #pragma omp parallel for
    for (int j = 0; j < output_size; ++j) {
        float b_grad = 0.0f;
        for (int b = 0; b < batches; ++b) {
            float g = grad_output(b, j);
            b_grad += g;

            for (int k = 0; k < input_size; ++k) {
                float dw = cached_input(b, k) * g;
                weights.grad[weights.get_index(k, j)] += dw; 
            }
        }
        biases.grad[j] += b_grad; 
    }

    return d_input;
}

void Dense::update_weights(float learning_rate) {
    for (int i = 0; i < weights.size(); ++i) {
        weights.data[i] -= learning_rate * weights.grad[i];
    }
    for (int i = 0; i < biases.size(); ++i) {
        biases.data[i] -= learning_rate * biases.grad[i];
    }
    weights.zero_grad();
    biases.zero_grad();
}

std::vector<Tensor*> Dense::get_parameters() {
    return {&weights, &biases};
}