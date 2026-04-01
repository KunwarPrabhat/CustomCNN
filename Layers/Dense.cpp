#include "Dense.h"

Dense::Dense(int input_size, int output_size)
    : weights(input_size, output_size),
      bias(1, output_size)
{
    weights.randomize();
    bias.randomize();
}

Tensor Dense::forward(const Tensor& input) {
    Tensor output(input.rows, weights.cols);

    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < weights.cols; j++) {
            output(i, j) = bias(0, j);

            for (int k = 0; k < input.cols; k++) {
                output(i, j) += input(i, k) * weights(k, j);
            }
        }
    }

    return output;
}