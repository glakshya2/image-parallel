#include "../../include/serial.hpp"

#include "../../include/stb_image.h"
#include "../../include/stb_image_write.h"

void comppressImageLossy(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, int outWidth, int outHeight, int scaleFactor) {
    for (int y = 0; y < outHeight; y++) {
        for (int x = 0; x < outWidth; x++) {
            for (int c = 0; c < channels; c++) {
                outputImage[(y * outWidth + x) * channels + c] = inputImage[(y * scaleFactor * width + x * scaleFactor) * channels + c];
            }
        }
    }
}