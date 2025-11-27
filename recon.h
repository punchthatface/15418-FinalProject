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

cv::Vec3b bilinearPredict(const cv::Mat& subsampled,
                                 const cv::Mat& mask,
                                 int y, int x,
                                 int factor);

void iterativeRefine(cv::Mat& img,
                     const cv::Mat& mask,
                     int iterations);

// Classify tiles using SAD on *subsampled* frames S_t and S_{t-1}
void classifyTilesSADSubsample(const cv::Mat& currSubsampled,
                               const cv::Mat& prevSubsampled,
                               int tileSize,
                               double sadThreshold,
                               cv::Mat& tileActiveMask);

// Iterative refinement that only does heavy work on active tiles
void iterativeRefineTiles(cv::Mat& img,
                          const cv::Mat& mask,
                          const cv::Mat& tileActiveMask,
                          int tileSize,
                          int iterations);