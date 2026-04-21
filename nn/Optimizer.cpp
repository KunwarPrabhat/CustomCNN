#include "Optimizer.h"

SGD::SGD(float lr) : learning_rate(lr) {}

void SGD::step(std::vector<std::shared_ptr<Layer>>& layers) {
    for (auto& layer : layers) {
        layer->update_weights(learning_rate);
    }
}
