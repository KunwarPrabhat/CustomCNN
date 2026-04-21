#include <iostream>
#include "nn/Model.h"
#include "Layers/Dense.h"
#include "Layers/Conv2D.h"
#include "Layers/MaxPool2D.h"
#include "Layers/Flatten.h"
#include "Layers/Dropout.h"
#include "Layers/BatchNorm2D.h"
#include "Layers/LeakyReLU.h"
#include "nn/Loss.h"
#include "nn/Adam.h"
#include "core/Tensor.h"
#include "data/Dataset.h"
#include "data/DataLoader.h"

int main() {
    std::cout << "--- MetalNet CNN Training (Phase 2 Utilities) ---" << std::endl;

    Model model;
    
    // First Block: Conv -> BatchNorm2D -> LeakyReLU -> Pool
    model << std::make_shared<Conv2D>(3, 8, 3, 1, 1) // Support 3 channel RGB input
          << std::make_shared<BatchNorm2D>(8)
          << std::make_shared<LeakyReLU>(0.01f)
          << std::make_shared<MaxPool2D>(2, 2);
    
    // Second Block: Conv -> BatchNorm2D -> LeakyReLU -> Pool
    model << std::make_shared<Conv2D>(8, 16, 3, 1, 1)
          << std::make_shared<BatchNorm2D>(16)
          << std::make_shared<LeakyReLU>(0.01f)
          << std::make_shared<MaxPool2D>(2, 2);
    
    model << std::make_shared<Flatten>();
    
    // Final Layers: Dense -> Dropout -> LeakyReLU -> Dense
    model << std::make_shared<Dense>(784, 64)
          << std::make_shared<Dropout>(0.5f)
          << std::make_shared<LeakyReLU>(0.01f)
          << std::make_shared<Dense>(64, 10);
    
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
            auto [batch_x, batch_y] = loader.next_batch();
            
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
    model2 << std::make_shared<Conv2D>(3, 8, 3, 1, 1) // Wait, input channels is 3 for RGB!
           << std::make_shared<BatchNorm2D>(8)
           << std::make_shared<LeakyReLU>(0.01f)
           << std::make_shared<MaxPool2D>(2, 2)
           << std::make_shared<Conv2D>(8, 16, 3, 1, 1) 
           << std::make_shared<BatchNorm2D>(16)
           << std::make_shared<LeakyReLU>(0.01f)
           << std::make_shared<MaxPool2D>(2, 2)
           << std::make_shared<Flatten>()
           << std::make_shared<Dense>(784, 64) 
           << std::make_shared<Dropout>(0.5f)
           << std::make_shared<LeakyReLU>(0.01f)
           << std::make_shared<Dense>(64, 10);

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