// recon_cuda.h
#pragma once

#include <opencv2/opencv.hpp>

// Initialize CUDA refinement and check if a CUDA device is available.
// Returns true iff a CUDA-capable device is found and usable.
bool cudaRefinementInit();

// GPU-backed version of:
//   iterativeRefine(img, mask, iterations)
//
// If no CUDA device is available, this function falls back to
// the CPU iterativeRefine() implementation.
void iterativeRefineCUDA(cv::Mat& img,
                         const cv::Mat& mask,
                         int iterations);

// GPU-backed version of:
//   iterativeRefineTiles(img, mask, tileActiveMask, tileSize, iterations)
//
// If no CUDA device is available, this function falls back to
// the CPU iterativeRefineTiles() implementation.
void iterativeRefineTilesCUDA(cv::Mat& img,
                              const cv::Mat& mask,
                              const cv::Mat& tileActiveMask,
                              int tileSize,
                              int iterations);

// Print accumulated CUDA timing stats (malloc/free/H2D/kernel/D2H).
// Safe to call even if CUDA was never used (it will just print zeros).
void cudaPrintRefineStats();

void reconstructTilesCUDA(cv::Mat& finalRecon,
                          const cv::Mat& prevRecon,
                          const cv::Mat& subsampled,
                          const cv::Mat& mask,
                          const cv::Mat& tileActiveMask,
                          int tileSize,
                          int factor);