#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <memory>

namespace MetalNet {

// [INIT FIX] Custom allocator that bypasses zero-initialization.
// This leaves RAM uninitialized, dropping allocation time from 200ms to ~1ms.
template <typename T>
struct NoInitAllocator : std::allocator<T> {
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        if constexpr (sizeof...(Args) == 0) {
            // Do nothing! 
        } else {
            ::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
        }
    }
};

struct QuantizedTensor {
  std::vector<int> shape;
  std::vector<int8_t> data;
  float scale = 1.0f;
  int32_t zero_point = 0;
};

class Tensor {
public:
  std::vector<int> shape;
  // Use the NoInitAllocator for instant OS-level allocation
  std::vector<float, NoInitAllocator<float>> data;
  std::vector<float, NoInitAllocator<float>> grad;

  inline Tensor() {}

  inline Tensor(int rows, int cols) : shape({rows, cols}) {
    data.resize(rows * cols); // No 0.0f passed! Triggers instant allocation
  }

  inline Tensor(int n, int c, int h, int w) : shape({n, c, h, w}) {
    data.resize(n * c * h * w);
  }

  inline explicit Tensor(std::vector<int> s) : shape(std::move(s)) {
    int total = 1;
    for (int d : shape) total *= d;
    data.resize(total);
  }

  inline void require_grad() {
    if (grad.size() != data.size()) {
      grad.resize(data.size(), 0.0f); // Gradients MUST still be zeroed
    }
  }

  inline void randomize() { fill_random(-0.01f, 0.01f); }

  inline void fill_random(float min = -1.0f, float max = 1.0f) {
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(min, max);
    for (auto &v : data) v = dis(gen);
  }

  inline void fill_he_init(int fan_in) {
    if (fan_in <= 0) return;
    static thread_local std::mt19937 gen(std::random_device{}());
    float limit = std::sqrt(2.0f / fan_in);
    std::normal_distribution<float> dis(0.0f, limit);
    for (auto &v : data) v = dis(gen);
  }

  inline void fill_constant(float value) { std::fill(data.begin(), data.end(), value); }
  inline void zero_grad() { if (!grad.empty()) std::fill(grad.begin(), grad.end(), 0.0f); }
  inline void fill(float v) { fill_constant(v); }

  inline int size() const { return (int)data.size(); }
  inline int dims() const { return (int)shape.size(); }

  inline int get_index(int i, int j) const { return i * shape[1] + j; }
  inline int get_index(int n, int c, int h, int w) const {
    return n * (shape[1] * shape[2] * shape[3]) + c * (shape[2] * shape[3]) + h * shape[3] + w;
  }
  inline int get_index(const std::vector<int> &idx) const {
    int i = 0, mul = 1;
    for (int d = (int)shape.size() - 1; d >= 0; --d) {
      i += idx[d] * mul;
      mul *= shape[d];
    }
    return i;
  }

  inline float &operator()(int i, int j) { return data[get_index(i, j)]; }
  inline float operator()(int i, int j) const { return data[get_index(i, j)]; }
  inline float &operator()(int n, int c, int h, int w) { return data[get_index(n, c, h, w)]; }
  inline float operator()(int n, int c, int h, int w) const { return data[get_index(n, c, h, w)]; }

  inline float *data_ptr() { return data.data(); }
  inline const float *data_ptr() const { return data.data(); }
  inline float *grad_ptr() { return grad.data(); }
  inline const float *grad_ptr() const { return grad.data(); }

  inline std::span<float> view() { return {data.data(), data.size()}; }
  inline std::span<const float> view() const { return {data.data(), data.size()}; }

  inline QuantizedTensor quantize() const {
    QuantizedTensor q;
    q.shape = shape;
    q.data.resize(data.size());
    if (data.empty()) return q;

    auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
    float mn = *min_it;
    float mx = *max_it;

    float sc = (mx - mn) / 255.0f;
    if (sc == 0.0f) sc = 1e-9f;

    int32_t zp = (int32_t)std::round(-128.0f - mn / sc);

    #pragma omp simd
    for (int i = 0; i < (int)data.size(); ++i) {
      int32_t qv = (int32_t)std::round(data[i] / sc) + zp;
      qv = std::max(-128, std::min(127, qv));
      q.data[i] = (int8_t)qv;
    }

    q.scale = sc;
    q.zero_point = zp;
    return q;
  }
};
} // namespace MetalNet