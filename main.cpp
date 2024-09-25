#define STBI_MAX_DIMENSIONS (1 << 30)
#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#include "include/serial.hpp"
#include "include/image_io.hpp"
#include "include/parallel.hpp"

#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
using namespace std;

int main() {
    int width, height, channels;
    const char* inputFilePath = "combined_image.jpg";
    const char* outputFilePathSerial = "output_image_serial.jpg";
    const char* outputFilePathCUDA = "output_image_cuda.jpg";

    // Load the image
    unsigned char* inputImage = loadImage(inputFilePath, width, height, channels);
    if (!inputImage) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    // Create an output image buffer for both methods
    unsigned char* outputImageSerial = new unsigned char[width * height * channels];
    unsigned char* outputImageCUDA = new unsigned char[width * height * channels];

    // Generate Gaussian kernel (modify kernel size and sigma as needed)
    int kernelSize = 5;
    float sigma = 1.0f;
    std::vector<std::vector<float>> kernel = generateGaussianKernel(kernelSize, sigma);

    // Measure and run the serial Gaussian blur
    auto startSerial = std::chrono::high_resolution_clock::now();
    applyGaussianBlur(inputImage, outputImageSerial, width, height, channels, kernel);
    auto endSerial = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> serialTime = endSerial - startSerial;
    std::cout << "Serial Gaussian Blur Time: " << serialTime.count() << " seconds" << std::endl;

    // Save the serial output image
    saveImage(outputFilePathSerial, outputImageSerial, width, height, channels);

    // Measure and run the CUDA Gaussian blur
    auto startCUDA = std::chrono::high_resolution_clock::now();
    applyGaussianBlurCUDAWrapper(inputImage, outputImageCUDA, width, height, channels, kernel);
    auto endCUDA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cudaTime = endCUDA - startCUDA;
    std::cout << "CUDA Gaussian Blur Time: " << cudaTime.count() << " seconds" << std::endl;

    // Save the CUDA output image
    saveImage(outputFilePathCUDA, outputImageCUDA, width, height, channels);

    // Free the memory
    stbi_image_free(inputImage);
    delete[] outputImageSerial;
    delete[] outputImageCUDA;

    std::cout << "Gaussian blur applied using both serial and CUDA. Output images saved." << std::endl;

    return 0;
}