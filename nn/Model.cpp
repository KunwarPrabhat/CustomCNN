#include "Model.h"

void Model::add(Layer* layer) {
    layers.push_back(layer);
}

Tensor Model::forward(const Tensor& input) {
    Tensor output = input;

    for (auto layer : layers) {
        output = layer->forward(output);
    }

    return output;
}