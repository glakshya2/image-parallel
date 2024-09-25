#include "../../include/parallel.hpp"

#include "../../include/stb_image.h"
#include "../../include/stb_image_write.h"

#include <cuda_runtime.h>

__global__ void downScaleKernel(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, int outWidth, int outHeight, int scaleFactor) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < outWidth && y < outHeight) {
        int inputIdx = (y * scaleFactor * width + x * scaleFactor) * channels;
        int outputIdx = (y * outWidth + x) * channels;

        for (int c = 0; c < channels; ++c) {
            outputImage[outputIdx + c] = inputImage[inputIdx + c];
        }
    }
}


void compressImageLossyCUDA(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, int outWidth, int outHeight, int scaleFactor) {
    unsigned char *d_inputImage, *d_outputImage;
    
    cudaMalloc((void**)&d_inputImage, width * height * channels * sizeof(unsigned char));
    cudaMalloc((void**)&d_outputImage, outWidth * outHeight * channels * sizeof(unsigned char));

    cudaMemcpy(d_inputImage, inputImage, width * height * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((outWidth + blockSize.x - 1) / blockSize.x, (outHeight + blockSize.y - 1) / blockSize.y);
    downScaleKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height, channels, outWidth, outHeight, scaleFactor);
    cudaDeviceSynchronize();
    cudaMemcpy(outputImage, d_outputImage, outWidth * outHeight * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}