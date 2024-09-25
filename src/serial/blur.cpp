#define M_PI 3.14159265358979323846

#include "../../include/serial.hpp"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#include "../../include/stb_image.h"
#include "../../include/stb_image_write.h"

// Generate Gaussian Kernel
std::vector<std::vector<float>> generateGaussianKernel(int kernelSize, float sigma) {
    std::vector<std::vector<float>> kernel(kernelSize, std::vector<float>(kernelSize));
    float sum = 0.0f;
    int halfSize = kernelSize / 2;
    float sigma2 = 2.0f * sigma * sigma;

    for (int i = -halfSize; i <= halfSize; i++) {
        for (int j = -halfSize; j <= halfSize; j++) {
            kernel[i + halfSize][j + halfSize] = exp(-(i * i + j * j) / sigma2) / (M_PI * sigma2);
            sum += kernel[i + halfSize][j + halfSize];
        }
    }

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

// Apply Gaussian Blur
void applyGaussianBlur(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, const std::vector<std::vector<float>>& kernel) {
    int kernelSize = kernel.size();
    int halfSize = kernelSize / 2;

    // Dynamically allocate sum based on the number of channels
    std::vector<float> sum(channels, 0.0f);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::fill(sum.begin(), sum.end(), 0.0f);  // Reset sum array for each pixel

            for (int ky = -halfSize; ky <= halfSize; ky++) {
                for (int kx = -halfSize; kx <= halfSize; kx++) {
                    int imageX = std::min(std::max(x + kx, 0), width - 1);
                    int imageY = std::min(std::max(y + ky, 0), height - 1);
                    int pixelIndex = (imageY * width + imageX) * channels;

                    for (int c = 0; c < channels; c++) {
                        sum[c] += inputImage[pixelIndex + c] * kernel[ky + halfSize][kx + halfSize];
                    }
                }
            }

            int outputIndex = (y * width + x) * channels;
            for (int c = 0; c < channels; c++) {
                outputImage[outputIndex + c] = std::min(std::max(int(sum[c]), 0), 255);
            }
        }
    }
}

