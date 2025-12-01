#pragma once
#include <opencv2/opencv.hpp>

// GPU version of the "active tiles" reconstruction stage.
// - subsampled: CV_8UC3
// - mask:       CV_8UC1 (0 = missing, 1 = known)
// - prevRecon:  CV_8UC3 (R_{t-1})
// - tileActive: CV_8UC1 (per-tile 0/1 mask)
// - tileSize:   size of each tile (e.g., 2, 4, 8, ...)
// - factor:     subsampling factor used in subsampleFrame()
void reconstructActiveTilesCUDA(const cv::Mat& subsampled,
                                const cv::Mat& mask,
                                const cv::Mat& prevRecon,
                                const cv::Mat& tileActive,
                                int tileSize,
                                int factor,
                                cv::Mat& finalRecon);
