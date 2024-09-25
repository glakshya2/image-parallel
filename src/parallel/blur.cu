#include "../../include/parallel.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>

__global__ void applyGaussianBlurCUDA(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, float* d_kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;  // Out-of-bounds check

    int halfSize = kernelSize / 2;
    int pixelIndex = (y * width + x) * channels;

    // Accumulators for each color channel
    float sum[4] = {0};  // Supports up to 4 channels (RGBA)

    for (int ky = -halfSize; ky <= halfSize; ky++) {
        for (int kx = -halfSize; kx <= halfSize; kx++) {
            int imageX = min(max(x + kx, 0), width - 1);
            int imageY = min(max(y + ky, 0), height - 1);
            int inputIndex = (imageY * width + imageX) * channels;

            float kernelVal = d_kernel[(ky + halfSize) * kernelSize + (kx + halfSize)];

            for (int c = 0; c < channels; c++) {
                sum[c] += inputImage[inputIndex + c] * kernelVal;
            }
        }
    }

    for (int c = 0; c < channels; c++) {
        outputImage[pixelIndex + c] = min(max(int(sum[c]), 0), 255);
    }
}

void applyGaussianBlurCUDAWrapper(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, const std::vector<std::vector<float>>& kernel) {
    // Flatten kernel into a 1D array
    int kernelSize = kernel.size();
    float* h_kernel = new float[kernelSize * kernelSize];
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            h_kernel[i * kernelSize + j] = kernel[i][j];
        }
    }

    // Allocate device memory
    unsigned char* d_inputImage;
    unsigned char* d_outputImage;
    float* d_kernel;

    cudaMalloc(&d_inputImage, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_outputImage, width * height * channels * sizeof(unsigned char));
    cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_inputImage, inputImage, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    applyGaussianBlurCUDA<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height, channels, d_kernel, kernelSize);

    // Copy result back to host
    cudaMemcpy(outputImage, d_outputImage, width * height * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    cudaFree(d_kernel);
    delete[] h_kernel;
}
