#include "../../include/parallel.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
__global__ void grayscaleKernel(unsigned char* d_inputImage, unsigned char* d_outputImage, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = d_inputImage[idx + 0];
        unsigned char g = d_inputImage[idx + 1];
        unsigned char b = d_inputImage[idx + 2];

        // Apply grayscale conversion using the luminance formula
        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);

        d_outputImage[idx + 0] = gray;
        d_outputImage[idx + 1] = gray;
        d_outputImage[idx + 2] = gray;

        if (channels == 4) {  // Preserve alpha channel if present
            d_outputImage[idx + 3] = d_inputImage[idx + 3];
        }
    }
}

void applyGrayscaleCUDA(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels) {
    unsigned char* d_inputImage = nullptr;
    unsigned char* d_outputImage = nullptr;
    size_t imageSize = width * height * channels * sizeof(unsigned char);

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_inputImage, imageSize);
    cudaMalloc((void**)&d_outputImage, imageSize);

    // Copy the image data from the host (CPU) to the device (GPU)
    cudaMemcpy(d_inputImage, inputImage, imageSize, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    auto start = std::chrono::high_resolution_clock::now();
    // Launch the kernel to perform grayscale conversion
    grayscaleKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height, channels);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Compute time: " << (std::chrono::duration<double>(end - start)).count() << " seconds" << std::endl; 

    // Copy the result from device back to host
    cudaMemcpy(outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    // Free the GPU memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}
