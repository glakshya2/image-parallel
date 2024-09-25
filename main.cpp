#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <cstring>  // For strcmp
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#include "include/serial.hpp"
#include "include/image_io.hpp"
#include "include/parallel.hpp"

void printUsage() {
    std::cout << "Usage: ./image-parallel <input_image> <operation>\n";
    std::cout << "Available operations: \n";
    std::cout << "  -blur <kernel size> : Apply Gaussian blur\n";
    std::cout << "  -gray : Convert to grayscale\n";
    // Add more operations here
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Error: Missing arguments.\n";
        printUsage();
        return -1;
    }

    const char* inputFilePath = argv[1];
    const char* operation = argv[2];

    int width, height, channels;

    // Load the image
    unsigned char* inputImage = loadImage(inputFilePath, width, height, channels);
    if (!inputImage) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    // Create output image buffers for serial and CUDA
    unsigned char* outputImageSerial = new unsigned char[width * height * channels];
    unsigned char* outputImageCUDA = new unsigned char[width * height * channels];



    if (strcmp(operation, "-blur") == 0) {
        if (argc < 4) {
            std::cerr << "Error: Missing arguments.\n";
            printUsage();
            return -1;
        }
        // Gaussian kernel setup
        int kernelSize = stoi(argv[3]);
        float sigma = 1.0f;
        std::vector<std::vector<float>> kernel = generateGaussianKernel(kernelSize, sigma);
        // Perform Gaussian blur using serial method
        auto startSerial = std::chrono::high_resolution_clock::now();
        applyGaussianBlur(inputImage, outputImageSerial, width, height, channels, kernel);
        auto endSerial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> serialTime = endSerial - startSerial;
        std::cout << "Serial Gaussian Blur Time: " << serialTime.count() << " seconds" << std::endl;

        // Save serial output image
        saveImage("output_image_serial.jpg", outputImageSerial, width, height, channels);

        // Perform Gaussian blur using CUDA method
        auto startCUDA = std::chrono::high_resolution_clock::now();
        applyGaussianBlurCUDAWrapper(inputImage, outputImageCUDA, width, height, channels, kernel);
        auto endCUDA = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cudaTime = endCUDA - startCUDA;
        std::cout << "CUDA Gaussian Blur Time: " << cudaTime.count() << " seconds" << std::endl;

        // Save CUDA output image
        saveImage("output_image_cuda.jpg", outputImageCUDA, width, height, channels);
    } else if (strcmp(operation, "-gray") == 0) {
        auto startSerial = std::chrono::high_resolution_clock::now();
        convertToGrayscale(inputImage, outputImageSerial, width, height, channels);
        auto endSerial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> serialTime = endSerial - startSerial;
        std::cout << "Serial Grayscale Time: " << serialTime.count() << " seconds" << std::endl;
        
        saveImage("output_image_serial.jpg", outputImageSerial, width, height, channels);

        auto startCUDA = std::chrono::high_resolution_clock::now();
        applyGrayscaleCUDA(inputImage, outputImageCUDA, width, height, channels);
        auto endCUDA = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cudaTime = endCUDA - startCUDA;
        std::cout << "CUDA Grayscale Time: " << cudaTime.count() << " seconds" << std::endl;

        saveImage("output_image_cuda.jpg", outputImageCUDA, width, height, channels);
    } else {
        std::cerr << "Error: Unknown operation '" << operation << "'\n";
        printUsage();
    }

    // Free memory
    stbi_image_free(inputImage);
    delete[] outputImageSerial;
    delete[] outputImageCUDA;

    return 0;
}