#include "include/MetalNet/MetalNet.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <omp.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
double get_peak_ram_mb() {
    PROCESS_MEMORY_COUNTERS pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
}
#else
double get_peak_ram_mb() { return 0.0; } 
#endif

using namespace MetalNet;

Model build_vgg_model() {
    Model model;
    model << conv2d(1, 64, 3, 1, 1) << batchnorm2d(64) << leaky_relu(0.01f)
          << conv2d(64, 64, 3, 1, 1) << batchnorm2d(64) << leaky_relu(0.01f)
          << maxpool2d(2, 2) 
          << conv2d(64, 128, 3, 1, 1) << batchnorm2d(128) << leaky_relu(0.01f)
          << conv2d(128, 128, 3, 1, 1) << batchnorm2d(128) << leaky_relu(0.01f)
          << maxpool2d(2, 2) 
          << flatten() << dense(128 * 7 * 7, 10);
    return model;
}
int main() {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    std::cout << "====================================================\n";
    std::cout << "  METALNET v1.0 - COMPREHENSIVE BENCHMARK SUITE\n";
    std::cout << "====================================================\n\n";

    std::cout << "[*] Loading MNIST Dataset...\n";
    Dataset full_ds = Dataset::load_csv("mnist_train.csv", true, 10, {1, 28, 28}, 0);
    full_ds.normalize();

    int total_samples = 2000;
    Tensor train_img(total_samples, 1, 28, 28);
    Tensor train_lbl(total_samples, 10);
    std::copy(full_ds.images.data.begin(), full_ds.images.data.begin() + train_img.size(), train_img.data.begin());
    std::copy(full_ds.labels.data.begin(), full_ds.labels.data.begin() + train_lbl.size(), train_lbl.data.begin());
    Dataset train_ds; train_ds.images = train_img; train_ds.labels = train_lbl;

    // Optional: Nuke the loaded dataset to keep RAM perfectly flat
    full_ds.images.data.clear(); full_ds.images.data.shrink_to_fit();
    full_ds.labels.data.clear(); full_ds.labels.data.shrink_to_fit();

    // ------------------------------------------------------------------
    // TEST 1: OPENMP THREAD SCALING
    // ------------------------------------------------------------------
    std::cout << "\n--- TEST 1: OPENMP THREAD SCALING (Batch 128) ---\n";
    {
        Model m = build_vgg_model();
        m.compile({128, 1, 28, 28});
        
        // METALNET IS NOW FULLY AUTONOMOUS! 
        // Calling eval() natively flushes all gradient memory for Inference.
        // If you ever want to train, just call m.train() and it restores it!
        m.eval(); 

        std::vector<int> thread_counts = {1, 2, 4, 8, 16};
        for (int t : thread_counts) {
            omp_set_num_threads(t);
            DataLoader loader(&train_ds, 128, false, false);
            
            auto start = std::chrono::high_resolution_clock::now();
            while(loader.has_next()) {
                auto [bx, by] = loader.next_batch();
                m.forward(bx);
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> dur = end - start;
            std::cout << "Threads: " << std::setw(2) << t << " | Throughput: " 
                      << std::fixed << std::setprecision(2) << (total_samples / dur.count()) << " img/sec\n";
        }
    }

    // ------------------------------------------------------------------
    // TEST 2: REAL-TIME INFERENCE LATENCY
    // ------------------------------------------------------------------
    std::cout << "\n--- TEST 2: INFERENCE LATENCY (Batch 1, 1 Thread) ---\n";
    {
        omp_set_num_threads(1); 
        Model inf_model = build_vgg_model();
        inf_model.compile({1, 1, 28, 28});
        inf_model.eval(); // Autonomously drops memory to minimum
        
        Tensor single_img(1, 1, 28, 28);
        for(int i=0; i<10; i++) inf_model.forward(single_img);

        auto inf_start = std::chrono::high_resolution_clock::now();
        int inf_runs = 500;
        for(int i=0; i<inf_runs; i++) inf_model.forward(single_img);
        std::chrono::duration<double, std::milli> inf_dur = std::chrono::high_resolution_clock::now() - inf_start;
        std::cout << "Avg Latency per image: " << (inf_dur.count() / inf_runs) << " ms\n";
    }

    // ------------------------------------------------------------------
    // TEST 3: PEAK MEMORY FOOTPRINT
    // ------------------------------------------------------------------
    std::cout << "\n--- TEST 3: PEAK RAM USAGE ---\n";
    std::cout << "Peak Working Set Size: " << get_peak_ram_mb() << " MB\n";
    
    std::cout << "\n====================================================\n";
    std::cout << "  BENCHMARK SUITE COMPLETE\n";
    std::cout << "====================================================\n";

    return 0;
}