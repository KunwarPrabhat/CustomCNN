#include "Concat.h"

Tensor Concat::forward(const std::vector<Tensor>& inputs) {
    if (inputs.empty()) throw std::runtime_error("Concat node expects strictly > 0 inputs");
    
    int total_c = 0;
    int n = inputs[0].shape[0];
    int h = inputs[0].shape[2];
    int w = inputs[0].shape[3];

    channel_splits.clear();
    for (const auto& in : inputs) {
        total_c += in.shape[1];
        channel_splits.push_back(in.shape[1]);
    }

    Tensor output(n, total_c, h, w);
    
    for (int b = 0; b < n; ++b) {
        int current_c = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            int c_in = inputs[i].shape[1];
            for (int c = 0; c < c_in; ++c) {
                // Highly cache-local tight copy
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        output(b, current_c + c, y, x) = inputs[i](b, c, y, x);
                    }
                }
            }
            current_c += c_in;
        }
    }
    return output;
}

std::vector<Tensor> Concat::backward_multi(const Tensor& grad_output) {
    std::vector<Tensor> grad_inputs;
    int n = grad_output.shape[0];
    int h = grad_output.shape[2];
    int w = grad_output.shape[3];

    int current_c = 0;
    for (int split : channel_splits) {
        Tensor grad_in(n, split, h, w);
        for (int b = 0; b < n; ++b) {
            for (int c = 0; c < split; ++c) {
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        grad_in(b, c, y, x) = grad_output(b, current_c + c, y, x);
                    }
                }
            }
        }
        current_c += split;
        grad_inputs.push_back(grad_in);
    }
    return grad_inputs;
}
