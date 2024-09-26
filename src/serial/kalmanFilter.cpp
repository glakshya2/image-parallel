#include "../../include/serial.hpp"

KalmanFilter::KalmanFilter(double init_x, double init_P, double A, double B, double H, double Q, double R)
    : X(init_x), P(init_P), A(A), B(B), H(H), Q(Q), R(R) {}

void KalmanFilter::predict(double control_input) {
    X = A * X + B * control_input;
    P = A * P * A + Q;
}

void KalmanFilter::update(double measurement) {
    double K = P * H / (H * P * H + R);
    X = X + K * (measurement - H * X);
    P = (1 - K * H) * P;
}

double KalmanFilter::getState() const {
    return X;
}

double KalmanFilter::getUncertainity() const {
    return P;
}

void applykalman(unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < channels; ++c) {
                int idx = (y * width + x) * channels + c;
                KalmanFilter kf(inputImage[idx], 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

                kf.update(inputImage[idx]);
                double filteredValue = kf.getState();
                outputImage[idx] = static_cast<unsigned char>(std::min(std::max(0.0, filteredValue), 255.0));
            }
        }
    }
}