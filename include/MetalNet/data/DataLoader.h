#pragma once
#include "Dataset.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace MetalNet {

class DataLoader {
public:
    Dataset*         dataset;
    int              batch_size;
    bool             shuffle;
    bool             drop_last;
    std::vector<int> indices;
    int              current_idx;
    std::mt19937     rng;
    Tensor           batch_x_A, batch_y_A;
    Tensor           batch_x_B, batch_y_B;
    bool             use_A;

    std::vector<bool> flips;
    std::vector<int>  shift_ys;
    std::vector<int>  shift_xs;
    bool             augment;

    // Async prefetching
    std::thread             prefetch_thread;
    std::mutex              mtx;
    std::condition_variable cv_produce;
    std::condition_variable cv_consume;
    bool                    stop_thread;
    bool                    buffer_B_ready;
    bool                    prefetch_active;

    inline DataLoader(Dataset* ds, int bs, bool sh=true, bool drop_last_=true, bool aug=false)
        : dataset(ds), batch_size(bs), shuffle(sh), drop_last(drop_last_), augment(aug), current_idx(0), rng(std::random_device{}()),
          use_A(true), stop_thread(false), buffer_B_ready(false), prefetch_active(false) {
        if (dataset && dataset->images.dims()>0) {
            int n=dataset->images.shape[0];
            indices.resize(n);
            for (int i=0;i<n;++i) indices[i]=i;
            
            // Output NHWC instead of NCHW
            int C=dataset->images.shape[1], H=dataset->images.shape[2], W=dataset->images.shape[3];
            int NC=dataset->labels.shape[1];
            
            batch_x_A = Tensor(batch_size, H, W, C);
            batch_y_A = Tensor(batch_size, NC);
            batch_x_B = Tensor(batch_size, H, W, C);
            batch_y_B = Tensor(batch_size, NC);

            flips.resize(batch_size);
            shift_ys.resize(batch_size);
            shift_xs.resize(batch_size);

            reset();

            // Start prefetch thread
            prefetch_active = true;
            prefetch_thread = std::thread(&DataLoader::prefetch_loop, this);
            
            // Prefetch first batch into buffer_B synchronously
            fill_buffer(batch_x_B, batch_y_B);
            buffer_B_ready = true;
        }
    }

    inline ~DataLoader() {
        if (prefetch_active) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                stop_thread = true;
            }
            cv_produce.notify_one();
            if (prefetch_thread.joinable()) {
                prefetch_thread.join();
            }
        }
    }

    inline bool has_next() const {
        if (!dataset || dataset->images.dims()==0) return false;
        if (drop_last) return current_idx + batch_size <= dataset->images.shape[0];
        return current_idx < dataset->images.shape[0];
    }

    inline void reset() {
        current_idx = 0;
        if (shuffle && !indices.empty()) {
            std::shuffle(indices.begin(), indices.end(), rng);
        }
        
        if (prefetch_active) {
            std::lock_guard<std::mutex> lock(mtx);
            buffer_B_ready = false;
        }
    }

    inline void fill_buffer(Tensor& bx, Tensor& by) {
        if (!has_next()) return;
        const int ns=dataset->images.shape[0];
        const int ei=std::min(current_idx+batch_size, ns);
        const int bs=ei-current_idx;
        const int C=dataset->images.shape[1], H=dataset->images.shape[2], W=dataset->images.shape[3];
        const int NC=dataset->labels.shape[1];
        
        bx.shape[0] = bs;
        by.shape[0] = bs;
        
        const float* src_x = dataset->images.data.data();
        const float* src_y = dataset->labels.data.data();
        float* dx = bx.data.data(); 
        float* dy = by.data.data();

        if (augment) {
            std::uniform_real_distribution<float> rand_flip(0.0f, 1.0f);
            std::uniform_int_distribution<int> rand_crop(-4, 4);
            for (int b=0; b<bs; ++b) {
                flips[b] = rand_flip(rng) > 0.5f;
                shift_ys[b] = rand_crop(rng);
                shift_xs[b] = rand_crop(rng);
            }
        }

        // Removed pragma omp parallel to avoid interference with master thread
        for (int b=0; b<bs; ++b) {
            int idx = indices[current_idx+b];
            const float* sx = src_x + idx*(C*H*W);
            float*       tx = dx    + b  *(H*W*C);
            
            if (augment) {
                bool flip = flips[b];
                int dy_shift = shift_ys[b];
                int dx_shift = shift_xs[b];
                
                for (int y=0; y<H; ++y) {
                    int sy = y - dy_shift;
                    for (int x=0; x<W; ++x) {
                        int sx_idx = x - dx_shift;
                        if (flip) sx_idx = (W - 1) - sx_idx;
                        
                        for (int c=0; c<C; ++c) {
                            float val = 0.0f;
                            if (sy >= 0 && sy < H && sx_idx >= 0 && sx_idx < W) {
                                val = sx[c * H * W + sy * W + sx_idx];
                            }
                            tx[y * W * C + x * C + c] = val; // NHWC
                        }
                    }
                }
            } else {
                for (int y=0; y<H; ++y) {
                    for (int x=0; x<W; ++x) {
                        for (int c=0; c<C; ++c) {
                            tx[y * W * C + x * C + c] = sx[c * H * W + y * W + x];
                        }
                    }
                }
            }

            const float* sy_ptr = src_y + idx*NC;
            float*       ty_ptr = dy    + b  *NC;
            for (int i=0; i<NC; ++i) ty_ptr[i] = sy_ptr[i];
        }
        
        current_idx = ei;
    }

    inline void prefetch_loop() {
        while (true) {
            std::unique_lock<std::mutex> lock(mtx);
            cv_produce.wait(lock, [this]() { return stop_thread || !buffer_B_ready; });
            
            if (stop_thread) break;
            
            if (has_next()) {
                Tensor* fill_x = use_A ? &batch_x_B : &batch_x_A;
                Tensor* fill_y = use_A ? &batch_y_B : &batch_y_A;
                fill_buffer(*fill_x, *fill_y);
            }
            
            buffer_B_ready = true;
            lock.unlock();
            cv_consume.notify_one();
        }
    }

    inline std::pair<Tensor&, Tensor&> next_batch() {
        std::unique_lock<std::mutex> lock(mtx);
        cv_consume.wait(lock, [this]() { return buffer_B_ready; });
        
        Tensor* out_x = use_A ? &batch_x_B : &batch_x_A;
        Tensor* out_y = use_A ? &batch_y_B : &batch_y_A;
        
        use_A = !use_A;
        buffer_B_ready = false;
        
        lock.unlock();
        cv_produce.notify_one();
        
        return std::pair<Tensor&, Tensor&>(*out_x, *out_y);
    }
};

} // namespace MetalNet
