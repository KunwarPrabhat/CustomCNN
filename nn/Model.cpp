#include "Model.h"
#include <fstream>
#include <stdexcept>

void Model::add_node(std::shared_ptr<Layer> node) {
    nodes.push_back(node);
}

void Model::connect(std::shared_ptr<Layer> from, std::shared_ptr<Layer> to) {
    from->output_nodes.push_back(to);
    to->input_nodes.push_back(from);
}

Model& Model::operator<<(std::shared_ptr<Layer> layer) {
    add_node(layer);
    if (nodes.size() > 1) {
        connect(nodes[nodes.size() - 2], layer);
    }
    return *this;
}

void Model::build_graph() {
    layers.clear();
    std::unordered_map<Layer*, int> in_degree;
    std::queue<std::shared_ptr<Layer>> queue;

    for (auto& node : nodes) {
        in_degree[node.get()] = node->input_nodes.size();
        if (in_degree[node.get()] == 0) {
            queue.push(node);
        }
    }

    while (!queue.empty()) {
        auto u = queue.front();
        queue.pop();
        layers.push_back(u);

        for (auto& v : u->output_nodes) {
            if (--in_degree[v.get()] == 0) {
                queue.push(v);
            }
        }
    }
}

Tensor Model::forward(const Tensor& input) {
    if (layers.empty()) build_graph();
    for (auto& layer : layers) {
        std::vector<Tensor> inputs_for_layer;
        if (layer->input_nodes.empty()) {
            inputs_for_layer.push_back(input);
        } else {
            for (auto& in_node : layer->input_nodes) {
                inputs_for_layer.push_back(in_node->output_cache);
            }
        }
        layer->output_cache = layer->forward(inputs_for_layer);
    }
    return layers.back()->output_cache;
}

void Model::backward(const Tensor& grad_output) {
    for (auto& layer : layers) {
        layer->has_grad_cache = false;
    }

    layers.back()->grad_cache = grad_output;
    layers.back()->has_grad_cache = true;

    for (int i = (int)layers.size() - 1; i >= 0; --i) {
        auto& layer = layers[i];
        std::vector<Tensor> grad_inputs = layer->backward_multi(layer->grad_cache);
        for (size_t j = 0; j < layer->input_nodes.size(); ++j) {
            auto& parent = layer->input_nodes[j];
            if (!parent->has_grad_cache) {
                parent->grad_cache = grad_inputs[j];
                parent->has_grad_cache = true;
            } else {
                for (int k = 0; k < parent->grad_cache.size(); ++k) {
                    parent->grad_cache.data[k] += grad_inputs[j].data[k];
                }
            }
        }
    }
}

void Model::train() {
    for (auto& layer : layers) layer->train();
}

void Model::eval() {
    for (auto& layer : layers) layer->eval();
}

void Model::save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    for (auto& layer : layers) {
        for (Tensor* state : layer->get_states()) {
            file.write(reinterpret_cast<const char*>(state->data.data()), state->data.size() * sizeof(float));
        }
    }
}

void Model::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    for (auto& layer : layers) {
        for (Tensor* state : layer->get_states()) {
            file.read(reinterpret_cast<char*>(state->data.data()), state->data.size() * sizeof(float));
        }
    }
}