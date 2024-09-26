#ifndef PARALLEL_HPP
#define PARALLEL_HPP

#include <vector>

using namespace std;

void applyGaussianBlur(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, const vector<vector<float>>& kernel);

void applyGaussianBlurCUDAWrapper(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, const std::vector<std::vector<float>>& kernel);

void applyGrayscaleCUDA(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels);

void compressImageLossyCUDA(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, int outWidth, int outHeight, int scaleFactor);

void contrastAdjustmentCUDA(unsigned char* inputImage, unsigned char* outputImage, float contrastFactor, int width, int height, int channels);

void applyKalmanFilterCUDA(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels);

#endif
