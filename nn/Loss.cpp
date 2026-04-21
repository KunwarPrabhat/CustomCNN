#include "Loss.h"
#include <cmath>
#include <algorithm>
#include <limits>

// MSELoss Implementation
float MSELoss::forward(const Tensor& preds, const Tensor& targets) {
    float loss = 0.0f;
    int N = preds.shape[0];
    int size = preds.size();

    #pragma omp parallel for reduction(+:loss)
    for (int i = 0; i < size; ++i) {
        float diff = preds.data[i] - targets.data[i];
        loss += diff * diff;
    }
    return loss / N;
}

Tensor MSELoss::backward(const Tensor& preds, const Tensor& targets) {
    Tensor d_preds(preds.shape);
    int N = preds.shape[0];
    int size = preds.size();

    for (int i = 0; i < size; ++i) {
        d_preds.data[i] = 2.0f * (preds.data[i] - targets.data[i]) / N;
    }
    return d_preds;
}

// CrossEntropyLoss Implementation (Includes Softmax activation implicitly)
float CrossEntropyLoss::forward(const Tensor& preds, const Tensor& targets) {
    int N = preds.shape[0];
    int C = preds.shape[1];
    float loss = 0.0f;

    #pragma omp parallel for reduction(+:loss)
    for (int i = 0; i < N; ++i) {
        // Find max for numeric stability
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < C; ++j) {
            max_val = std::max(max_val, preds(i, j));
        }

        // Sum exponentiations
        float sum_exp = 0.0f;
        for (int j = 0; j < C; ++j) {
            sum_exp += std::exp(preds(i, j) - max_val);
        }

        // Cross-entropy sum over classes
        for (int j = 0; j < C; ++j) {
            // Target is normally 1 for correct class, 0 for others
            if (targets(i, j) > 0.0f) {
                float prob = std::exp(preds(i, j) - max_val) / sum_exp;
                // Add epsilon for log zero safety
                loss -= targets(i, j) * std::log(prob + 1e-9f);
            }
        }
    }
    return loss / N;
}

Tensor CrossEntropyLoss::backward(const Tensor& preds, const Tensor& targets) {
    int N = preds.shape[0];
    int C = preds.shape[1];
    Tensor d_preds(preds.shape);

    for (int i = 0; i < N; ++i) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < C; ++j) {
            max_val = std::max(max_val, preds(i, j));
        }

        float sum_exp = 0.0f;
        for (int j = 0; j < C; ++j) {
            sum_exp += std::exp(preds(i, j) - max_val);
        }

        for (int j = 0; j < C; ++j) {
            float prob = std::exp(preds(i, j) - max_val) / sum_exp;
            d_preds(i, j) = (prob - targets(i, j)) / N;
        }
    }
    return d_preds;
}
