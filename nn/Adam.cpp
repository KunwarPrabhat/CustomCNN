#include "Adam.h"
#include <cmath>

Adam::Adam(float lr, float b1, float b2, float eps) 
    : learning_rate(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {}

void Adam::step(std::vector<std::shared_ptr<Layer>>& layers) {
    t++;
    for (auto& layer : layers) {
        std::vector<Tensor*> params = layer->get_parameters();
        for (Tensor* param : params) {
            // Initialize moments if empty
            if (m.find(param) == m.end()) {
                m[param] = Tensor(param->shape);
                v[param] = Tensor(param->shape);
                m[param].fill(0.0f);
                v[param].fill(0.0f);
            }

            Tensor& m_t = m[param];
            Tensor& v_t = v[param];

            for (int i = 0; i < param->size(); ++i) {
                float grad = param->grad[i];
                
                // Update biased first moment estimate
                m_t.data[i] = beta1 * m_t.data[i] + (1.0f - beta1) * grad;
                
                // Update biased second raw moment estimate
                v_t.data[i] = beta2 * v_t.data[i] + (1.0f - beta2) * grad * grad;
                
                // Compute bias-corrected first moment estimate
                float m_hat = m_t.data[i] / (1.0f - std::pow(beta1, t));
                
                // Compute bias-corrected second raw moment estimate
                float v_hat = v_t.data[i] / (1.0f - std::pow(beta2, t));
                
                // Update parameters
                param->data[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
            
            // Clear gradients
            param->zero_grad();
        }
    }
}
