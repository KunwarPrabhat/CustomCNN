#include "ElementwiseAdd.h"

Tensor ElementwiseAdd::forward(const std::vector<Tensor>& inputs) {
    if (inputs.size() < 2) throw std::runtime_error("ElementwiseAdd node requires >= 2 inputs for geometric merge");
    
    Tensor output = inputs[0]; // Deep allocation base
    
    for (size_t i = 1; i < inputs.size(); ++i) {
        const Tensor& in = inputs[i];
        for (int j = 0; j < output.size(); ++j) {
            output.data[j] += in.data[j];
        }
    }
    return output;
}

std::vector<Tensor> ElementwiseAdd::backward_multi(const Tensor& grad_output) {
    std::vector<Tensor> grad_inputs;
    
    // An addition node passes total gradients to every connected underlying input identically
    for (size_t i = 0; i < input_nodes.size(); ++i) {
        grad_inputs.push_back(grad_output); 
    }
    
    return grad_inputs;
}
