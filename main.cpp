#include "include/MetalNet/MetalNet.h"
#include <iostream>
using namespace MetalNet;

int main() {
    std::cout << "--- MetalNet CNN Training (Phase 2 Utilities) ---" << std::endl;

    Model model;
    
    // First Block: Conv -> BatchNorm2D -> LeakyReLU -> Pool
    model << conv2d(3, 8, 3, 1, 1) // Support 3 channel RGB input
          << batchnorm2d(8)
          << leaky_relu(0.01f)
          << maxpool2d(2, 2);
    
    // Second Block: Conv -> BatchNorm2D -> LeakyReLU -> Pool
    model << conv2d(8, 16, 3, 1, 1)
          << batchnorm2d(16)
          << leaky_relu(0.01f)
          << maxpool2d(2, 2);
    
    model << flatten();
    
    // Final Layers: Dense -> Dropout -> LeakyReLU -> Dense
    model << dense(784, 64)
          << dropout(0.5f)
          << leaky_relu(0.01f)
          << dense(64, 10);
    
    CrossEntropyLoss criterion;
    Adam optimizer(0.001f);
    
    // Create a dummy Dataset of 16 RGB images (16, 3, 28, 28)
    Dataset ds;
    ds.images = Tensor(16, 3, 28, 28);
    ds.images.randomize();
    ds.labels = Tensor(16, 10);
    ds.labels.fill(0.0f);
    for(int i=0; i<16; ++i) {
        ds.labels(i, i % 10) = 1.0f; // Mock continuous labels
    }
    
    DataLoader loader(&ds, 4, true); // batch size 4, shuffle = true

    std::cout << "Starting Training loop with DataLoader..." << std::endl;
    model.train(); 

    for(int epoch = 1; epoch <= 3; epoch++) {
        loader.reset();
        int batch_idx = 0;
        
        while (loader.has_next()) {
            auto batch = loader.next_batch();
            Tensor& batch_x = batch.first;
            Tensor& batch_y = batch.second;
            
            Tensor preds = model.forward(batch_x);
            float loss = criterion.forward(preds, batch_y);
            
            Tensor grad_out = criterion.backward(preds, batch_y);
            model.backward(grad_out);
            optimizer.step(model.layers);
            
            std::cout << "Epoch " << epoch << ", Batch " << ++batch_idx << " | Loss: " << loss << std::endl;
        }
    }
    
    model.eval(); 
    std::cout << "Saving Model to test_model.bin..." << std::endl;
    model.save("test_model.bin");

    // Test Serialization by loading into a new identical model structure
    Model model2;
    model2 << conv2d(3, 8, 3, 1, 1) // Wait, input channels is 3 for RGB!
           << batchnorm2d(8)
           << leaky_relu(0.01f)
           << maxpool2d(2, 2)
           << conv2d(8, 16, 3, 1, 1) 
           << batchnorm2d(16)
           << leaky_relu(0.01f)
           << maxpool2d(2, 2)
           << flatten()
           << dense(784, 64) 
           << dropout(0.5f)
           << leaky_relu(0.01f)
           << dense(64, 10);

    std::cout << "Loading Model from test_model.bin..." << std::endl;
    model2.load("test_model.bin");
    model2.eval();

    // Verify
    Tensor test_x(1, 3, 28, 28);
    test_x.randomize();
    Tensor out1 = model.forward(test_x);
    Tensor out2 = model2.forward(test_x);
    
    float diff = 0.0f;
    for(int i=0; i<10; ++i) diff += std::abs(out1.data[i] - out2.data[i]);
    
    std::cout << "Difference between Model 1 and Loaded Model 2: " << diff << std::endl;
    if (diff < 1e-5f) std::cout << "Serialization Success!" << std::endl;

    std::cout << "Done! Everything looks good." << std::endl;
    return 0;
}