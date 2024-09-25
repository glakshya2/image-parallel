#include "../../include/serial.hpp"

void convertToGrayscale(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels) {
    for (int i = 0; i < width * height; ++i) {
        int r = inputImage[i * channels + 0];  // Red
        int g = inputImage[i * channels + 1];  // Green
        int b = inputImage[i * channels + 2];  // Blue
        
        // Grayscale value using the luminance formula
        unsigned char gray = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        
        // Set all channels to the grayscale value (assuming output is RGB, otherwise just one channel if grayscale)
        outputImage[i * channels + 0] = gray;
        outputImage[i * channels + 1] = gray;
        outputImage[i * channels + 2] = gray;

        if (channels == 4) {  // Handle alpha channel if present
            outputImage[i * channels + 3] = inputImage[i * channels + 3];
        }
    }
}
