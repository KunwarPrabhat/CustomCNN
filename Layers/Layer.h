#pragma once
#include "../core/Tensor.h"
#include <vector>
#include <memory>
#include <stdexcept>

class Layer {
public:
    virtual ~Layer() = default;

    std::vector<std::shared_ptr<Layer>> input_nodes;
    std::vector<std::shared_ptr<Layer>> output_nodes;

    Tensor output_cache;
    Tensor grad_cache;
    Tensor cached_input;
    bool has_grad_cache = false;

    virtual Tensor forward(const Tensor& input) {
        throw std::runtime_error("Single-input forward not implemented for this layer.");
    }
    
    virtual Tensor backward(const Tensor& grad_output) {
        throw std::runtime_error("Single-input backward not implemented for this layer.");
    }

    virtual Tensor forward(const std::vector<Tensor>& inputs) {
        if (inputs.size() == 1) return forward(inputs[0]);
        throw std::runtime_error("This layer does not support multiple topological inputs.");
    }

    virtual std::vector<Tensor> backward_multi(const Tensor& grad_output) {
        return {backward(grad_output)};
    }
    
    virtual void update_weights(float learning_rate) {}
    
    bool is_training = true;
    virtual void train() { is_training = true; }
    virtual void eval() { is_training = false; }
    
    virtual std::vector<Tensor*> get_parameters() { return {}; }
    virtual std::vector<Tensor*> get_states() { return get_parameters(); }
};