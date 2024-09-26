#include "../../include/parallel.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void applyKalmanFilter(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, double A, double B, double H, double Q, double R){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int pixelCount = width * height * channels;
    if (idx < pixelCount) {
        double x = inputImage[idx];
        double P = 1.0;

        for (int i = 0; i < channels; i++){
            double K = P * H /(H * P * H + R);
            x = x + K * (inputImage[idx] - H * x);
            P = (1 - K * H) * P;

            outputImage[idx] = static_cast<unsigned char>(min(max(0.0, x), 255.0));        
        }
    }
}

void applyKalmanFilterCUDA(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels) {
    unsigned char *d_inputImage, *d_outputImage;
    size_t size = width * height * channels * sizeof(unsigned char);

    cudaMalloc(&d_inputImage, size);
    cudaMalloc(&d_outputImage, size);

    cudaMemcpy(d_inputImage, inputImage, size, cudaMemcpyHostToDevice);
    
    double A = 1.0;
    double B = 0.0;
    double H = 1.0;
    double Q = 0.1;
    double R = 0.1;

    int blockSize = 256;
    int numBlocks = (width * height * channels + blockSize - 1) / blockSize;

    auto start = std::chrono::high_resolution_clock::now();
    applyKalmanFilter<<<numBlocks, blockSize>>>(d_inputImage, d_outputImage, width, height, channels, A, B, H, Q, R);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "GPU Compute time: " << (std::chrono::duration<double>(end - start)).count() << " seconds" << std::endl; 
    cudaMemcpy(outputImage, d_outputImage, size, cudaMemcpyDeviceToHost);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}