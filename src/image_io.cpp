#include "../include/image_io.hpp"
#include <iostream>
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"

// Load Image using stb_image
unsigned char* loadImage(const char* filePath, int &width, int &height, int &channels) {
    std::cout << "Attempting to load image from: " << filePath << std::endl;
    unsigned char* image = stbi_load(filePath, &width, &height, &channels, 0);
    if (!image) {
        std::cerr << "Error: Could not load image " << filePath << " " << stbi_failure_reason() << std::endl;
        return nullptr;
    }
    return image;
}

// Save Image using stb_image_write
void saveImage(const char* filePath, unsigned char* image, int width, int height, int channels) {
    if (!stbi_write_jpg(filePath, width, height, channels, image, 100)) {
        std::cerr << "Error: Could not write image " << filePath << std::endl;
    }
}
