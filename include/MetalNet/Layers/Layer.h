#pragma once
#include "../core/Tensor.h"
#include <vector>
#include <memory>
#include <stdexcept>

namespace MetalNet {

class Layer {
public:
    virtual ~Layer() = default;

    std::vector<std::shared_ptr<Layer>> input_nodes;
    std::vector<std::shared_ptr<Layer>> output_nodes;
    std::vector<std::shared_ptr<Layer>> extra_input_nodes;
    std::vector<Tensor> extra_inputs;
    Tensor output_cache, grad_cache, cached_input;
    bool   has_grad_cache = false;
    bool   is_training    = true;

    inline void add_extra_input_source(std::shared_ptr<Layer> parent) {
        extra_input_nodes.push_back(parent);
    }

    virtual Tensor forward(const Tensor& input) {
        throw std::runtime_error("forward(Tensor) not implemented");
    }
    virtual Tensor backward(const Tensor& grad) {
        throw std::runtime_error("backward(Tensor) not implemented");
    }
    virtual Tensor forward(const std::vector<Tensor>& inputs) {
        if (inputs.size() == 1) return forward(inputs[0]);
        throw std::runtime_error("Multi-input forward not implemented");
    }
    virtual std::vector<Tensor> backward_multi(const Tensor& grad) {
        return {backward(grad)};
    }
    virtual void update_weights(float lr) {}
    virtual void train() { is_training = true; }
    virtual void eval()  { is_training = false; }
    virtual std::vector<Tensor*> get_parameters() { return {}; }
    virtual std::vector<Tensor*> get_states()     { return get_parameters(); }
};

} // namespace MetalNet
