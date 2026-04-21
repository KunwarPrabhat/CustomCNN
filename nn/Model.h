#pragma once
#include <vector>
#include <memory>
#include <unordered_map>
#include <queue>
#include "../Layers/Layer.h"
#include "Optimizer.h"
#include "Loss.h"

class Model {
public:
    std::vector<std::shared_ptr<Layer>> nodes;
    std::vector<std::shared_ptr<Layer>> layers;

    void add_node(std::shared_ptr<Layer> node);
    void connect(std::shared_ptr<Layer> from, std::shared_ptr<Layer> to);
    Model& operator<<(std::shared_ptr<Layer> layer); // Compatibility
    void build_graph();

    Tensor forward(const Tensor& input);
    void backward(const Tensor& grad_output);

    void train();
    void eval();

    void save(const std::string& filename);
    void load(const std::string& filename);
};