#include "metrics.h"
#include <cmath>
#include <stdexcept>

double computeMSE(const cv::Mat& a, const cv::Mat& b) {
    if (a.size() != b.size() || a.type() != b.type()) {
        throw std::runtime_error("computeMSE: mismatched frames");
    }

    double mse = 0.0;
    int rows = a.rows;
    int cols = a.cols;

    int channels = a.channels();
    long long count = 0;

    for (int y = 0; y < rows; ++y) {
        const uchar* pa = a.ptr<uchar>(y);
        const uchar* pb = b.ptr<uchar>(y);
        for (int x = 0; x < cols * channels; ++x) {
            double diff = static_cast<double>(pa[x]) - static_cast<double>(pb[x]);
            mse += diff * diff;
            count++;
        }
    }

    return mse / static_cast<double>(count);
}

double mseToPSNR(double mse) {
    if (mse <= 1e-10) return 100.0;  // effectively perfect
    double maxI = 255.0;
    return 10.0 * std::log10((maxI * maxI) / mse);
}

double computeSSIM(const cv::Mat& a, const cv::Mat& b) {
    // Basic checks
    if (a.empty() || b.empty()) {
        throw std::runtime_error("computeSSIM: one or both images are empty");
    }
    if (a.size() != b.size()) {
        throw std::runtime_error("computeSSIM: size mismatch");
    }
    if (a.type() != b.type()) {
        throw std::runtime_error("computeSSIM: type mismatch");
    }

    cv::Mat i1, i2;

    // Convert to grayscale float [0,255]
    if (a.channels() == 3) {
        cv::cvtColor(a, i1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(b, i2, cv::COLOR_BGR2GRAY);
    } else {
        i1 = a.clone();
        i2 = b.clone();
    }

    i1.convertTo(i1, CV_32F);
    i2.convertTo(i2, CV_32F);

    // Constants for SSIM (standard values for 8-bit images)
    const double C1 = (0.01 * 255.0) * (0.01 * 255.0);
    const double C2 = (0.03 * 255.0) * (0.03 * 255.0);

    // Gaussian window
    int winSize = 11;
    double sigma = 1.5;

    cv::Mat mu1, mu2;
    cv::GaussianBlur(i1, mu1, cv::Size(winSize, winSize), sigma);
    cv::GaussianBlur(i2, mu2, cv::Size(winSize, winSize), sigma);

    cv::Mat mu1_sq   = mu1.mul(mu1);
    cv::Mat mu2_sq   = mu2.mul(mu2);
    cv::Mat mu1_mu2  = mu1.mul(mu2);

    cv::Mat sigma1_sq, sigma2_sq, sigma12;

    cv::GaussianBlur(i1.mul(i1), sigma1_sq, cv::Size(winSize, winSize), sigma);
    sigma1_sq -= mu1_sq;

    cv::GaussianBlur(i2.mul(i2), sigma2_sq, cv::Size(winSize, winSize), sigma);
    sigma2_sq -= mu2_sq;

    cv::GaussianBlur(i1.mul(i2), sigma12, cv::Size(winSize, winSize), sigma);
    sigma12 -= mu1_mu2;

    // SSIM formula
    cv::Mat t1 = 2.0 * mu1_mu2 + C1;
    cv::Mat t2 = 2.0 * sigma12 + C2;
    cv::Mat t3 = mu1_sq + mu2_sq + C1;
    cv::Mat t4 = sigma1_sq + sigma2_sq + C2;

    cv::Mat ssim_map;
    cv::divide(t1.mul(t2), t3.mul(t4), ssim_map);

    // Mean SSIM over image
    cv::Scalar mssim = cv::mean(ssim_map);
    return mssim[0];  // single-channel, so use [0]
}