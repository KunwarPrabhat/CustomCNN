#pragma once
#include <vector>
#include <memory>
#include <unordered_map>
#include <queue>
#include <fstream>
#include "../Layers/Layer.h"
#include "Optimizer.h"
#include "Loss.h"

namespace MetalNet {

class Model {
public:
    std::vector<std::shared_ptr<Layer>> nodes;
    std::vector<std::shared_ptr<Layer>> layers;

    inline void add_node(std::shared_ptr<Layer> n) {
        nodes.push_back(n);
    }

    inline void connect(std::shared_ptr<Layer> from, std::shared_ptr<Layer> to) {
        from->output_nodes.push_back(to);
        to->input_nodes.push_back(from);
    }

    // Fluent sequential builder: model << layerA << layerB
    inline Model& operator<<(std::shared_ptr<Layer> layer) {
        add_node(layer);
        if (nodes.size() > 1)
            connect(nodes[nodes.size() - 2], layer);
        return *this;
    }

    inline void build_graph() {
        layers.clear();
        std::unordered_map<Layer*, int> in_degree;
        std::queue<std::shared_ptr<Layer>> q;

        for (auto& n : nodes) {
            in_degree[n.get()] = (int)n->input_nodes.size();
            if (in_degree[n.get()] == 0) q.push(n);
        }
        while (!q.empty()) {
            auto u = q.front(); q.pop();
            layers.push_back(u);
            for (auto& v : u->output_nodes)
                if (--in_degree[v.get()] == 0) q.push(v);
        }
    }

    inline void add_residual_link(std::shared_ptr<Layer> parent, std::shared_ptr<Layer> child) {
        child->add_extra_input_source(parent);
    }

    inline Tensor forward(const Tensor& input) {
        if (layers.empty()) build_graph();
        for (auto& layer : layers) {
            std::vector<Tensor> ins;
            if (layer->input_nodes.empty()) {
                ins.push_back(input);
            } else {
                for (auto& n : layer->input_nodes)
                    ins.push_back(n->output_cache);
            }
            
            layer->extra_inputs.clear();
            for (auto& en : layer->extra_input_nodes) {
                layer->extra_inputs.push_back(en->output_cache);
            }

            if (!layer->extra_inputs.empty() && !ins.empty()) {
                Tensor combined_input = ins[0];
                for (const auto& ei : layer->extra_inputs) {
                    float* d = combined_input.data.data();
                    const float* s = ei.data.data();
                    int sz = combined_input.size();
                    #pragma omp simd
                    for (int i=0; i<sz; ++i) d[i] += s[i];
                }
                ins[0] = combined_input;
            }
            
            layer->output_cache = layer->forward(ins);
        }
        return layers.back()->output_cache;
    }

    inline void backward(const Tensor& grad_output) {
        for (auto& layer : layers) layer->has_grad_cache = false;
        layers.back()->grad_cache    = grad_output;
        layers.back()->has_grad_cache = true;

        for (int i = (int)layers.size() - 1; i >= 0; --i) {
            auto& layer  = layers[i];
            auto  grads  = layer->backward_multi(layer->grad_cache);
            for (size_t j = 0; j < layer->input_nodes.size(); ++j) {
                auto& parent = layer->input_nodes[j];
                if (!parent->has_grad_cache) {
                    parent->grad_cache    = grads[j];
                    parent->has_grad_cache = true;
                } else {
                    // Accumulate gradients from multiple downstream paths (skip connections)
                    float*       dst = parent->grad_cache.data.data();
                    const float* src = grads[j].data.data();
                    int          sz  = parent->grad_cache.size();
                    #pragma omp simd
                    for (int k = 0; k < sz; ++k) dst[k] += src[k];
                }
            }
            for (size_t j = 0; j < layer->extra_input_nodes.size(); ++j) {
                auto& extra_parent = layer->extra_input_nodes[j];
                if (!extra_parent->has_grad_cache) {
                    extra_parent->grad_cache    = grads[0];
                    extra_parent->has_grad_cache = true;
                } else {
                    float*       dst = extra_parent->grad_cache.data.data();
                    const float* src = grads[0].data.data();
                    int          sz  = extra_parent->grad_cache.size();
                    #pragma omp simd
                    for (int k = 0; k < sz; ++k) dst[k] += src[k];
                }
            }
        }
    }

    inline void train() { for (auto& l : layers) l->train(); }
    inline void eval()  { for (auto& l : layers) l->eval();  }

    inline void save(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        for (auto& layer : layers)
            for (Tensor* s : layer->get_states())
                file.write(reinterpret_cast<const char*>(s->data.data()),
                           s->data.size() * sizeof(float));
    }

    inline void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        for (auto& layer : layers)
            for (Tensor* s : layer->get_states())
                file.read(reinterpret_cast<char*>(s->data.data()),
                          s->data.size() * sizeof(float));
    }
};

} // namespace MetalNet
