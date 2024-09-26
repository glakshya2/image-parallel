#ifndef SERIAL_HPP
#define SERIAL_HPP

#include <vector>

using namespace std;

vector<vector<float>> generateGaussianKernel(int kernelSize, float sigma);

void applyGaussianBlur(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, const vector<vector<float>>& kernel);

void convertToGrayscale(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels);

void comppressImageLossy(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, int outWidth, int outHeight, int scaleFactor);

void adjustImageContrast(unsigned char* inputImage, unsigned char* outputImage, float contrastFactor, int width, int height, int channels);

class KalmanFilter {
    public:
    KalmanFilter(double init_X, double init_P, double A, double B, double H, double Q, double R);
    void predict(double control_input);
    void update(double measurment);

    double getState() const;
    double getUncertainity() const;

    private:
    double X;
    double P;

    double A;
    double B;
    double H;
    double Q;
    double R; 
};

void applykalman(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels);

#endif