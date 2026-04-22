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

    inline void compile(const std::vector<int>& primary_input_shape) {
        if (layers.empty()) build_graph();
        for (auto& layer : layers) {
            std::vector<std::vector<int>> input_shapes;
            if (layer->input_nodes.empty()) {
                input_shapes.push_back(primary_input_shape);
            } else {
                for (auto& p : layer->input_nodes)
                    input_shapes.push_back(p->output_buffer.shape);
            }
            for (auto& p : layer->extra_input_nodes) {
                input_shapes.push_back(p->output_buffer.shape);
            }
            
            layer->compile(input_shapes);

            if (!layer->extra_input_nodes.empty()) {
                layer->temp_add_buffer = Tensor(input_shapes[0]);
            }
            layer->grad_output_buffer = Tensor(layer->output_buffer.shape);
        }
    }

    inline void forward(const Tensor& input) {
        if (layers.empty()) return;
        for (auto& layer : layers) {
            std::vector<const Tensor*> ins;
            if (layer->input_nodes.empty()) {
                ins.push_back(&input);
            } else {
                for (auto& n : layer->input_nodes)
                    ins.push_back(&n->output_buffer);
            }
            
            layer->extra_inputs.clear();
            for (auto& en : layer->extra_input_nodes) {
                layer->extra_inputs.push_back(&en->output_buffer);
            }

            if (!layer->extra_inputs.empty() && !ins.empty()) {
                const Tensor* primary = ins[0];
                Tensor& comb = layer->temp_add_buffer;
                float* d = comb.data.data();
                const float* p = primary->data.data();
                int sz = comb.size();
                #pragma omp simd
                for (int i=0; i<sz; ++i) d[i] = p[i];

                for (const auto* ei : layer->extra_inputs) {
                    const float* s = ei->data.data();
                    #pragma omp simd
                    for (int i=0; i<sz; ++i) d[i] += s[i];
                }
                ins[0] = &comb;
            }
            
            layer->forward(ins);
        }
    }

    inline void backward(const Tensor& grad_output) {
        for (auto& layer : layers) layer->has_grad_cache = false;
        
        float* dst = layers.back()->grad_output_buffer.data.data();
        const float* src = grad_output.data.data();
        int sz = grad_output.size();
        #pragma omp simd
        for (int k=0; k<sz; ++k) dst[k] = src[k];
        layers.back()->has_grad_cache = true;

        for (int i = (int)layers.size() - 1; i >= 0; --i) {
            auto& layer  = layers[i];
            
            layer->backward_multi(layer->grad_output_buffer);

            for (size_t j = 0; j < layer->input_nodes.size(); ++j) {
                auto& parent = layer->input_nodes[j];
                const Tensor& grad_to_pass = (layer->input_nodes.size() == 1) ? layer->grad_input_buffer : layer->multi_grad_input_buffers[j];

                if (!parent->has_grad_cache) {
                    float* pdst = parent->grad_output_buffer.data.data();
                    const float* psrc = grad_to_pass.data.data();
                    int psz = parent->grad_output_buffer.size();
                    #pragma omp simd
                    for (int k = 0; k < psz; ++k) pdst[k] = psrc[k];
                    parent->has_grad_cache = true;
                } else {
                    float* pdst = parent->grad_output_buffer.data.data();
                    const float* psrc = grad_to_pass.data.data();
                    int psz = parent->grad_output_buffer.size();
                    #pragma omp simd
                    for (int k = 0; k < psz; ++k) pdst[k] += psrc[k];
                }
            }

            for (size_t j = 0; j < layer->extra_input_nodes.size(); ++j) {
                auto& extra_parent = layer->extra_input_nodes[j];
                const Tensor& grad_to_pass = layer->grad_input_buffer;

                if (!extra_parent->has_grad_cache) {
                    float* pdst = extra_parent->grad_output_buffer.data.data();
                    const float* psrc = grad_to_pass.data.data();
                    int psz = extra_parent->grad_output_buffer.size();
                    #pragma omp simd
                    for (int k = 0; k < psz; ++k) pdst[k] = psrc[k];
                    extra_parent->has_grad_cache = true;
                } else {
                    float* pdst = extra_parent->grad_output_buffer.data.data();
                    const float* psrc = grad_to_pass.data.data();
                    int psz = extra_parent->grad_output_buffer.size();
                    #pragma omp simd
                    for (int k = 0; k < psz; ++k) pdst[k] += psrc[k];
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
