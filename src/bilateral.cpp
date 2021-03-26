#include "bilateral.hpp"
#include "cmath"
float weights[1000];
void BilateralFilter::run()
{
    omp_set_dynamic(0);
    omp_set_num_threads(4);
    unsigned int i;
    unsigned int j;
    unsigned int k;
#pragma omp parallel for private(i) private(j) private(k)
    for (i = 0; i < height; ++i) {
        for (j = 0; j < width; ++j) {
            for (k = 0; k < 3; ++k) {
                newImage[4 * width * i + 4 * j + k] =
                    newColor(i, j, k);
            }
            newImage[4 * width * i + 4 * j + 3] =
                oldImage[4 * width * i + 4 * j + 3];
        }
    }
}

float BilateralFilter::w(unsigned int row1, unsigned int column1,
                         unsigned int row2, unsigned int column2,
                         unsigned int i)
{
    return 1.f / (exp(pow(width * row2 + column2 - width * row1 - column1, 2) *
                      1.f / (2 * pow(SYGMA1, 2))) *
                  exp(pow(oldImage[4 * width * row2 + 4 * column2 + i] -
                              oldImage[4 * width * row1 + 4 * column1 + i],
                          2) *
                      1.f / (2 * pow(SYGMA2, 2))));
}
float BilateralFilter::C(unsigned int row, unsigned int column, unsigned int i)
{
    float resultValue = 0;
    unsigned int currWeightCounter = 0;
    float currWeight = 0;
    for (int j = int(row) - RADIUS; j <= int(row) + RADIUS; ++j) {  // row
        for (int k = int(column) - RADIUS; k <= int(column) + RADIUS;
             ++k) {  // num in row
            if ((j < 0) || (j >= height) || (k < 0) || (k >= width)) {
                continue;
            }
            currWeight = w(row, column, (unsigned int)j, (unsigned int)k, i);
            resultValue += currWeight;
            weights[currWeightCounter] = currWeight;
            currWeightCounter++;
        }
    }

    return resultValue;
}

float BilateralFilter::newColor(unsigned int row, unsigned int column,
                                unsigned int i)
{
    float newColor;
    float c;
    uint currWeightCounter = 0;
    c = C(row, column, i);
    currWeightCounter = 0;
    newColor = 0.0;
    for (int j = int(row) - RADIUS; j <= int(row) + RADIUS; ++j) {  // row
        for (int k = int(column) - RADIUS; k <= int(column) + RADIUS;
             ++k) {  // num in row
            if ((j < 0) || (j >= height) || (k < 0) || (k >= width)) {
                continue;
            }
            else {
                newColor += oldImage[4 * width * uint(j) + 4 * uint(k) + i] *
                            weights[currWeightCounter] / c;
                currWeightCounter++;
            }
        }
    }
    return newColor;
}