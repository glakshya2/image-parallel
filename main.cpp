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
    std::cout << "Usage: ./image-parallel <input_image> <operation>" << std::endl;
    std::cout << "Available operations: " << std::endl;
    std::cout << "  -blur <Kernel Size> : Apply Gaussian blur" << std::endl;
    std::cout << "  -gray : Convert to grayscale" << std::endl;
    std::cout << "  -compress <Scale Factor> " << std::endl;
    std::cout << "  -contrast <Contrast Factor> " << std::endl;
    std::cout << "  -kalman" << std::endl;
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
    
    int argFour = 0;
    if (argc == 4) {
        argFour = stoi(argv[3]);
    }
    int outWidth = width, outHeight = height;
    if (strcmp(operation, "-compress") == 0) {
        outWidth = width / argFour;
        outHeight = height / argFour;
    }

    // Create output image buffers for serial and CUDA
    unsigned char* outputImageSerial = new unsigned char[outWidth * outHeight * channels];
    unsigned char* outputImageCUDA = new unsigned char[outWidth * outHeight * channels];



    if (strcmp(operation, "-blur") == 0) {
        if (argc < 4) {
            std::cerr << "Error: Missing arguments.\n";
            printUsage();
            return -1;
        }
        // Gaussian kernel setup
        int kernelSize = argFour;
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
    } else if (strcmp(operation, "-compress") == 0) {
        int scaleFactor = argFour;
        auto startSerial = std::chrono::high_resolution_clock::now();
        comppressImageLossy(inputImage, outputImageSerial, width, height, channels, outWidth, outHeight, scaleFactor);
        auto endSerial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> serialTime = endSerial - startSerial;
        std::cout << "Serial Compression Time: " << serialTime.count() << " seconds" << std::endl;

        saveImage("output_image_serial.jpg", outputImageSerial, outWidth, outHeight, channels); 

        auto startCUDA = std::chrono::high_resolution_clock::now();
        compressImageLossyCUDA(inputImage, outputImageCUDA, width, height, channels, outWidth, outHeight, scaleFactor);
        auto endCUDA = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cudaTime = endCUDA - startCUDA;
        std::cout << "CUDA Compression Time: " << cudaTime.count() << " seconds" << std::endl;

        saveImage("output_image_cuda.jpg", outputImageCUDA, outWidth, outHeight, channels);
    } else if (strcmp(operation, "-contrast") == 0) {
        float contrastFactor = static_cast<float> (argFour);
        auto startSerial = std::chrono::high_resolution_clock::now();
        adjustImageContrast(inputImage, outputImageSerial, contrastFactor, width, height, channels);
        auto endSerial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> serialTime = endSerial - startSerial;
        std::cout << "Serial Contrast Time: " << serialTime.count() << std::endl;

        saveImage("output_image_serial.jpg", outputImageSerial, outWidth, outHeight, channels);

        auto startCUDA = std::chrono::high_resolution_clock::now();
        contrastAdjustmentCUDA(inputImage, outputImageCUDA, contrastFactor, width, height, channels);
        auto endCUDA = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> CUDATime = endCUDA - startCUDA;
        std::cout << "CUDA Contrast Time: " << CUDATime.count() << " seconds" << std::endl;

        saveImage("output_image_cuda.jpg", outputImageCUDA, width, height, channels);
    } else if (strcmp(operation, "-kalman") == 0) {
        auto startSerial = std::chrono::high_resolution_clock::now();
        applykalman(inputImage, outputImageSerial, width, height, channels);
        auto endSerial = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> serialTime = endSerial - startSerial;
        std::cout << "Serial Kalman Time: " << serialTime.count() << std::endl;

        saveImage("output_image_serial.jpg", outputImageSerial, width, height, channels);

        auto startCUDA = std::chrono::high_resolution_clock::now();
        applyKalmanFilterCUDA(inputImage, outputImageCUDA, width, height, channels);
        auto endCUDA = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> CUDATime = endCUDA - startCUDA;
        std::cout << "Parallel Kalman Time: " << CUDATime.count() << std::endl;

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