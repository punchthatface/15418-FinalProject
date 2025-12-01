#pragma once
#include <opencv2/opencv.hpp>

double computeMSE(const cv::Mat& a, const cv::Mat& b);
double mseToPSNR(double mse);
double computeSSIM(const cv::Mat& a, const cv::Mat& b);