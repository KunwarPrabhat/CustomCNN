#pragma once
#include <vector>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <span>
#include <random>

namespace MetalNet {

struct QuantizedTensor {
    std::vector<int>    shape;
    std::vector<int8_t> data;
    float    scale      = 1.0f;
    int32_t  zero_point = 0;
};

class Tensor {
public:
    std::vector<int>   shape;
    std::vector<float> data;
    std::vector<float> grad;

    inline Tensor() {}
    inline Tensor(int rows, int cols) : shape({rows, cols}) {
        data.resize(rows * cols, 0.0f);
        grad.resize(rows * cols, 0.0f);
    }
    inline Tensor(int n, int c, int h, int w) : shape({n, c, h, w}) {
        data.resize(n * c * h * w, 0.0f);
        grad.resize(n * c * h * w, 0.0f);
    }
    inline explicit Tensor(std::vector<int> s) : shape(std::move(s)) {
        int total = 1;
        for (int d : shape) total *= d;
        data.resize(total, 0.0f);
        grad.resize(total, 0.0f);
    }

    inline void randomize() {
        for (auto& v : data) v = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * 0.01f;
    }
    inline void fill_random(float min, float max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min, max);
        for (auto& v : data) v = dis(gen);
    }
    inline void fill_constant(float value) {
        std::fill(data.begin(), data.end(), value);
    }
    inline void zero_grad() { std::fill(grad.begin(), grad.end(), 0.0f); }
    inline void fill(float v) { std::fill(data.begin(), data.end(), v); }
    inline int  size() const { return (int)data.size(); }
    inline int  dims() const { return (int)shape.size(); }

    // Linear index helpers
    inline int get_index(int i, int j) const { return i * shape[1] + j; }
    inline int get_index(int n, int c, int h, int w) const {
        return n * (shape[1]*shape[2]*shape[3]) + c * (shape[2]*shape[3]) + h * shape[3] + w;
    }
    inline int get_index(const std::vector<int>& idx) const {
        int i = 0, mul = 1;
        for (int d = (int)shape.size()-1; d >= 0; --d) { i += idx[d]*mul; mul *= shape[d]; }
        return i;
    }

    // Accessors
    inline float& operator()(int i, int j)               { return data[get_index(i,j)]; }
    inline float  operator()(int i, int j) const         { return data[get_index(i,j)]; }
    inline float& operator()(int n, int c, int h, int w) { return data[get_index(n,c,h,w)]; }
    inline float  operator()(int n, int c, int h, int w) const { return data[get_index(n,c,h,w)]; }

    // std::span views
    inline std::span<float>       view()       { return {data.data(), data.size()}; }
    inline std::span<const float> view() const { return {data.data(), data.size()}; }

    // int8 quantization
    inline QuantizedTensor quantize() const {
        QuantizedTensor q;
        q.shape = shape;
        q.data.resize(data.size());
        if (data.empty()) return q;
        float mn = *std::min_element(data.begin(), data.end());
        float mx = *std::max_element(data.begin(), data.end());
        float sc = (mx - mn) / 255.0f;
        if (sc == 0.0f) sc = 1e-9f;
        int32_t zp = (int32_t)std::round(-128.0f - mn / sc);
        for (int i = 0; i < (int)data.size(); ++i) {
            int32_t qv = (int32_t)std::round(data[i] / sc) + zp;
            qv = std::max(-128, std::min(127, qv));
            q.data[i] = (int8_t)qv;
        }
        q.scale      = sc;
        q.zero_point = zp;
        return q;
    }
};

} // namespace MetalNet
