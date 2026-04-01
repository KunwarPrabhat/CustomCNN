#include <iostream>
#include "nn/Model.h"
#include "layers/Dense.h"
#include "layers/ReLU.h"

int main() {
    Model model;

    model.add(new Dense(3, 5));
    model.add(new ReLU());
    model.add(new Dense(5, 2));

    Tensor input(1, 3);
    input.randomize();

    Tensor output = model.forward(input);

    for (int i = 0; i < output.cols; i++) {
        std::cout << output(0, i) << " ";
    }

    return 0;
}