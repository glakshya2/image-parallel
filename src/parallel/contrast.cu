#include "../../include/parallel.hpp"

#include "../../include/stb_image.h"
#include "../../include/stb_image_write.h"

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__device__ unsigned char clampCUDA(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return static_cast<unsigned char> (value);
}

__global__ void contrastAdjustKernel(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, float contrastFactor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalPixels = width * height * channels;
    if (idx < totalPixels) {
        int pixelValue = inputImage[idx];
        int adjustedValue = static_cast<int> ((pixelValue - 128) * contrastFactor + 128);
        outputImage[idx] = clampCUDA(adjustedValue);
    }
}

void contrastAdjustmentCUDA(unsigned char* inputImage, unsigned char* outputImage, float contrastFactor, int width, int height, int channels) {
    unsigned char *d_inputImage, *d_outputImage;

    int totalPixels = width * height * channels;

    cudaMalloc((void**) &d_inputImage, (size_t) totalPixels * sizeof(unsigned char));
    cudaMalloc((void**) &d_outputImage, (size_t) totalPixels * sizeof(unsigned char));

    cudaMemcpy(d_inputImage, inputImage, (size_t) totalPixels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (totalPixels + blockSize - 1) / blockSize;
    auto start = std::chrono::high_resolution_clock::now();
    contrastAdjustKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height, channels, contrastFactor);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Compute time: " << (std::chrono::duration<double>(end - start)).count() << " seconds" << std::endl; 
    cudaMemcpy(outputImage, d_outputImage, (size_t) totalPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}