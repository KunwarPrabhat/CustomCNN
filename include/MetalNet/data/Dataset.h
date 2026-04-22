#pragma once
#include <string>
#include <fstream>
#include <stdexcept>
#include "../core/Tensor.h"

namespace MetalNet {

class Dataset {
public:
    Tensor images, labels;

    static inline void skipComments(std::ifstream& f) {
        char c; f>>c;
        while (c=='#') { f.ignore(256,'\n'); f>>c; }
        f.unget();
    }

    static inline Tensor load_ppm_grayscale(const std::string& fp) {
        std::ifstream f(fp, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open: "+fp);
        std::string m; f>>m;
        if (m!="P6"&&m!="P3") throw std::runtime_error("Bad PPM: "+m);
        skipComments(f);
        int W,H,mc; f>>W>>H>>mc; f.get();
        Tensor img(1,1,H,W);
        float* d=img.data.data();
        for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
            float r,g,b;
            if (m=="P6"){unsigned char px[3];f.read((char*)px,3);r=px[0];g=px[1];b=px[2];}
            else{int ri,gi,bi;f>>ri>>gi>>bi;r=ri;g=gi;b=bi;}
            d[y*W+x]=0.299f*r+0.587f*g+0.114f*b;
        }
        return img;
    }

    static inline Tensor load_ppm_rgb(const std::string& fp) {
        std::ifstream f(fp, std::ios::binary);
        if (!f) throw std::runtime_error("Cannot open: "+fp);
        std::string m; f>>m;
        if (m!="P6"&&m!="P3") throw std::runtime_error("Bad PPM: "+m);
        skipComments(f);
        int W,H,mc; f>>W>>H>>mc; f.get();
        Tensor img(1,3,H,W);
        float* r_=img.data.data();
        float* g_=r_+H*W;
        float* b_=g_+H*W;
        for (int y=0;y<H;++y) for (int x=0;x<W;++x) {
            float r,g,b;
            if (m=="P6"){unsigned char px[3];f.read((char*)px,3);r=px[0];g=px[1];b=px[2];}
            else{int ri,gi,bi;f>>ri>>gi>>bi;r=ri;g=gi;b=bi;}
            r_[y*W+x]=r; g_[y*W+x]=g; b_[y*W+x]=b;
        }
        return img;
    }

    inline void normalize() {
        float* d=images.data.data();
        #pragma omp simd
        for (int i=0;i<images.size();++i) d[i]/=255.0f;
    }
};

} // namespace MetalNet
