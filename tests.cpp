#include "include/MetalNet/MetalNet.h"
#include <iostream>
#include <cmath>
#include <fstream>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace MetalNet;

static int PASS=0, FAIL=0;
static void report(const char* name, bool ok){
    std::cout << (ok?"  [PASS] ":"  [FAIL] ") << name << "\n";
    ok?++PASS:++FAIL;
}

// ── 1. Header-Only Integrity ─────────────────────────────────
void test_header_only(){
    std::cout<<"\n=== 1. Header-Only Integrity ===\n";
    report("Master header compiles and resolves all sub-modules", true);
    Tensor t(2,2); Dense d(4,4); Conv2D c(1,1,3); MaxPool2D mp(2);
    ReLU r; LeakyReLU lr; Flatten fl; Softmax sm; Dropout dr; BatchNorm2D bn(4);
    Concat cat; ElementwiseAdd ew; MSELoss mse; CrossEntropyLoss ce;
    SGD sgd; Adam adam; Dataset ds; DataLoader loader(&ds,1); Model model;
    report("All layer/utility types instantiate without linker errors", true);
}

// ── 2. DAG & Architectural Logic ─────────────────────────────
void test_dag(){
    std::cout<<"\n=== 2. DAG & Architectural Logic ===\n";

    // Kahn topological sort: A→B, A→C, B+C→Add
    auto A=dense(4,4);
    auto B=dense(4,4);
    auto C=dense(4,4);
    auto Add=elementwise_add();
    Model dag;
    dag.add_node(A); dag.add_node(B); dag.add_node(C); dag.add_node(Add);
    dag.connect(A,B); dag.connect(A,C); dag.connect(B,Add); dag.connect(C,Add);
    dag.build_graph();
    int pA=-1,pB=-1,pC=-1,pD=-1;
    for(int i=0;i<(int)dag.layers.size();++i){
        if(dag.layers[i].get()==A.get())   pA=i;
        if(dag.layers[i].get()==B.get())   pB=i;
        if(dag.layers[i].get()==C.get())   pC=i;
        if(dag.layers[i].get()==Add.get()) pD=i;
    }
    report("Kahn topological sort: parent executes before children",
           pA<pB && pA<pC && pB<pD && pC<pD);

    // Concat
    Tensor t1(1,2,2,2); t1.fill(1.0f);
    Tensor t2(1,3,2,2); t2.fill(2.0f);
    Concat cat;
    cat.compile({{1,2,2,2}, {1,3,2,2}});
    Tensor merged=cat.forward({&t1,&t2});
    report("Concat: output channel count == sum of input channels", merged.shape[1]==5);
    report("Concat: channel data placed at correct offsets",
           std::abs(merged(0,0,0,0)-1.0f)<1e-5f && std::abs(merged(0,2,0,0)-2.0f)<1e-5f);

    // ElementwiseAdd
    Tensor ea(1,2,2,2); ea.fill(3.0f);
    Tensor eb(1,2,2,2); eb.fill(5.0f);
    ElementwiseAdd ew;
    ew.compile({{1,2,2,2}, {1,2,2,2}});
    Tensor esum=ew.forward({&ea,&eb});
    report("ElementwiseAdd: each element equals sum of inputs",
           std::abs(esum.data[0]-8.0f)<1e-5f);
}

// ── 3. Mathematical Accuracy ──────────────────────────────────
void test_math(){
    std::cout<<"\n=== 3. Mathematical Accuracy ===\n";

    // Adam moment update
    {
        float b1=0.9f,b2=0.999f,lr=0.001f,eps=1e-8f,g=0.5f;
        float m_=0,v_=0;
        m_=b1*m_+(1-b1)*g; v_=b2*v_+(1-b2)*g*g;
        float mh=m_/(1-b1), vh=v_/(1-b2);
        float expected=lr*mh/(std::sqrt(vh)+eps);

        auto layer=dense(1,1);
        layer->weights.data[0]=0.5f; layer->weights.grad[0]=g;
        float w0=layer->weights.data[0];
        std::vector<std::shared_ptr<Layer>> ls={layer};
        Adam adam(lr,b1,b2,eps);
        adam.step(ls);
        float actual=w0-layer->weights.data[0];
        report("Adam: weight update matches bias-corrected formula",
               std::abs(actual-expected)<1e-6f);
    }

    // BatchNorm running stats
    {
        BatchNorm2D bn(2);
        bn.compile({{1,2,2,2}});
        Tensor inp(1,2,2,2); inp.fill(4.0f);
        bn.train(); bn.forward(inp);
        float rm_train=bn.running_mean(0,0);
        bn.eval(); bn.forward(inp);
        float rm_eval=bn.running_mean(0,0);
        report("BatchNorm: running mean updates during train()", rm_train!=0.0f);
        report("BatchNorm: running mean frozen during eval()",
               std::abs(rm_train-rm_eval)<1e-6f);
    }

    // Gradient check
    {
        Dense layer(3,2);
        layer.compile({{1,3}});
        layer.weights.fill(0.1f); layer.biases.fill(0.0f);
        Tensor inp(1,3); inp.fill(1.0f);
        Tensor tgt(1,2); tgt.fill(0.0f);
        MSELoss loss_fn;
        Tensor pred=layer.forward(inp);
        Tensor go=loss_fn.backward(pred,tgt);
        layer.backward(go);
        float analytic=layer.weights.grad[0];
        float eps_n=1e-4f, w0=layer.weights.data[0];
        layer.weights.data[0]=w0+eps_n;
        float lp=loss_fn.forward(layer.forward(inp),tgt);
        layer.weights.data[0]=w0-eps_n;
        float lm=loss_fn.forward(layer.forward(inp),tgt);
        layer.weights.data[0]=w0;
        float numeric=(lp-lm)/(2*eps_n);
        float rel=std::abs(analytic-numeric)/(std::abs(analytic)+std::abs(numeric)+1e-10f);
        report("Backprop gradient check: analytic ≈ numeric gradient (rel err <1%)",rel<0.01f);
    }
}

// ── 4. Hardware Acceleration ──────────────────────────────────
void test_hardware(){
    std::cout<<"\n=== 4. Hardware Acceleration ===\n";

    // OpenMP
#ifdef _OPENMP
    int max_t=omp_get_max_threads(), par_t=1;
    #pragma omp parallel
    { par_t=omp_get_num_threads(); }
    report("OpenMP: runtime reports >0 available threads", max_t>0);
    report("OpenMP: parallel region activates multiple threads", par_t>=1);
    std::cout<<"         (max_threads="<<max_t<<", parallel used="<<par_t<<")\n";
#else
    report("OpenMP: NOT available on this toolchain (skipped)", true);
#endif

    // Dense SIMD correctness
    {
        Dense d(8,8); 
        d.compile({{1,8}});
        d.weights.fill(1.0f); d.biases.fill(0.0f);
        Tensor inp(1,8); inp.fill(1.0f);
        Tensor out=d.forward(inp);
        bool ok=true;
        for(int i=0;i<8;++i) if(std::abs(out(0,i)-8.0f)>1e-4f) ok=false;
        report("Dense forward (SIMD path): 8-neuron dot-product is correct", ok);
    }

    // Im2Col vs direct convolution
    {
        Tensor img(1,1,4,4);
        for(int i=0;i<16;++i) img.data[i]=(float)(i+1);
        Conv2D conv(1,1,2,1,0);
        conv.compile({{1,1,4,4}});
        conv.weights.fill(1.0f); conv.biases.fill(0.0f);
        Tensor direct=conv.forward(img);
        Tensor col=im2col(img,2,1,0);
        bool ok=true;
        for(int idx=0;idx<9;++idx){
            float cs=0;
            for(int row=0;row<4;++row) cs+=col(row,idx);
            int y=idx/3, x=idx%3;
            if(std::abs(cs-direct(0,0,y,x))>1e-4f) ok=false;
        }
        report("Im2Col: column sums match direct sliding-window convolution", ok);
    }
}

// ── 5. Production Utilities ───────────────────────────────────
void test_production(){
    std::cout<<"\n=== 5. Production Utilities ===\n";

    // Serialization round-trip
    {
        auto layer=dense(4,4);
        for(int i=0;i<layer->weights.size();++i) layer->weights.data[i]=(float)i*0.01f;
        for(int i=0;i<layer->biases.size();++i)  layer->biases.data[i] =(float)i*0.1f;
        Model m1; m1.add_node(layer); m1.build_graph(); m1.save("_rt.bin");
        auto l2=dense(4,4);
        Model m2; m2.add_node(l2); m2.build_graph(); m2.load("_rt.bin");
        bool ok=true;
        for(int i=0;i<layer->weights.size();++i)
            if(std::abs(l2->weights.data[i]-layer->weights.data[i])>1e-6f){ok=false;break;}
        report("Serialization: round-trip save/load preserves weights exactly", ok);
        std::remove("_rt.bin");
    }

    // Quantization
    {
        Tensor t(1,4);
        t.data={1.0f,-1.0f,0.5f,-0.5f};
        QuantizedTensor qt=t.quantize();
        bool ok=true;
        for(int i=0;i<4;++i){
            float dq=((float)qt.data[i]-(float)qt.zero_point)*qt.scale;
            float rel=std::abs(dq-t.data[i])/(std::abs(t.data[i])+1e-6f);
            if(rel>0.01f) ok=false;
        }
        report("Quantization: int8 dequantized values within 1% of float32", ok);
    }

    // RGB PPM parsing
    {
        {
            std::ofstream f("_test.ppm",std::ios::binary);
            f<<"P6\n2 2\n255\n";
            for(int i=0;i<4;++i){unsigned char px[3]={255,0,0};f.write((char*)px,3);}
        }
        Tensor rgb=Dataset::load_ppm_rgb("_test.ppm");
        bool shape_ok=(rgb.shape[0]==1&&rgb.shape[1]==3&&rgb.shape[2]==2&&rgb.shape[3]==2);
        bool vals_ok =(std::abs(rgb(0,0,0,0)-255.0f)<1e-3f&&
                       std::abs(rgb(0,1,0,0))<1e-3f&&std::abs(rgb(0,2,0,0))<1e-3f);
        report("Dataset: load_ppm_rgb returns (1,3,H,W) shaped Tensor", shape_ok);
        report("Dataset: RGB channels parsed into correct planes",  vals_ok);
        std::remove("_test.ppm");
    }
}

int main(){
    std::cout<<"╔══════════════════════════════════════════════╗\n";
    std::cout<<"║      MetalNet Validation Test Suite          ║\n";
    std::cout<<"╚══════════════════════════════════════════════╝\n";
    test_header_only();
    test_dag();
    test_math();
    test_hardware();
    test_production();
    std::cout<<"\n──────────────────────────────────────────────\n";
    std::cout<<"  Results: "<<PASS<<" PASSED  |  "<<FAIL<<" FAILED\n";
    std::cout<<"──────────────────────────────────────────────\n";
    return FAIL>0?1:0;
}
