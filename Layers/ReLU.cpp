#include "ReLU.h"

Tensor ReLU::forward(const Tensor& input) {
    Tensor output(input.rows, input.cols);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            output(i, j) = input.data[i * input.cols + j] > 0
                         ? input.data[i * input.cols + j]
                         : 0;
        }
    }

    return output;
}