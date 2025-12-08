#pragma once

#include <opencv2/opencv.hpp>

// Initialize CUDA refinement (returns false if no CUDA device)
bool cudaRefinementInit();

// Full-frame iterative refinement on GPU
void iterativeRefineCUDA(cv::Mat& img,
                         const cv::Mat& mask,
                         int iterations);

// Tile-aware iterative refinement on GPU
void iterativeRefineTilesCUDA(cv::Mat& img,
                              const cv::Mat& mask,
                              const cv::Mat& tileActiveMask,
                              int tileSize,
                              int iterations);

// Tile-based reconstruction on GPU
void reconstructTilesCUDA(cv::Mat& out,
                          const cv::Mat& prevRecon,
                          const cv::Mat& subsampled,
                          const cv::Mat& mask,
                          const cv::Mat& tileActiveMask,
                          int tileSize,
                          int factor);

// NEW: Tile classification on GPU using SAD over subsampled frames
void classifyTilesSADSubsampleCUDA(const cv::Mat& currSubsampled,
                                   const cv::Mat& prevSubsampled,
                                   int tileSize,
                                   double sadThreshold,
                                   cv::Mat& tileActiveMask);

// Print accumulated CUDA stats to stderr
void cudaPrintRefineStats();
