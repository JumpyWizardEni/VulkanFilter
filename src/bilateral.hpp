#define SYGMA1 35
#define SYGMA2 35
#define RADIUS 10
#include <iostream>
#include <omp.h>


class BilateralFilter {
    unsigned int width;
    unsigned int height;

public:
   
    float *oldImage;
    float *newImage;
    BilateralFilter(float *oldIm, float *newIm, unsigned int width_, unsigned int height_): oldImage(oldIm), newImage(newIm), width(width_), height(height_) {};
    void run();
    float w(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int);
    float C(unsigned int, unsigned int, unsigned int);
    float newColor(unsigned int, unsigned int, unsigned int);
};