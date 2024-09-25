#ifndef IMAGE_IO_HPP
#define IMAGE_IO_HPP

// Function declarations for image I/O
unsigned char* loadImage(const char* filePath, int &width, int &height, int &channels);
void saveImage(const char* filePath, unsigned char* image, int width, int height, int channels);

#endif // IMAGE_IO_HPP
