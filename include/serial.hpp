#ifndef SERIAL_HPP
#define SERIAL_HPP

#include <vector>

using namespace std;

vector<vector<float>> generateGaussianKernel(int kernelSize, float sigma);

void applyGaussianBlur(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, const vector<vector<float>>& kernel);

void convertToGrayscale(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels);

void comppressImageLossy(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, int outWidth, int outHeight, int scaleFactor);

#endif