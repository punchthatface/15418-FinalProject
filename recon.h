#pragma once
#include <opencv2/opencv.hpp>

void subsampleFrame(const cv::Mat& input,
                    cv::Mat& subsampled,
                    cv::Mat& mask,
                    int factor);

void reconstructNaive(const cv::Mat& subsampled,
                      const cv::Mat& mask,
                      int factor,
                      cv::Mat& output);

void temporalRefine(const cv::Mat& spatialCurrent,
                    const cv::Mat& prevRecon,
                    float alpha,
                    cv::Mat& refined);

void iterativeRefine(cv::Mat& img,
                     const cv::Mat& mask,
                     int iterations);