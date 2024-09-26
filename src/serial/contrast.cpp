#include "../../include/serial.hpp"

// #include "../../include/stb_image.h"
// #include "../../include/stb_image_write.h"

#include <cmath>

unsigned char clamp(int value) {
    if (value < 0) return 0;
    if (value > 255) return 255;
    return static_cast<unsigned char> (value);
}

void adjustImageContrast(unsigned char* inputImage, unsigned char* outputImage, float contrastFactor, int width, int height, int channels) {
    for (int i = 0; i < width * height * channels; i++) {
        int pixelValue = inputImage[i];
        int adjustedValue = static_cast<int> ((pixelValue - 128) * contrastFactor + 128);
        outputImage[i] = clamp(adjustedValue);
    }
}