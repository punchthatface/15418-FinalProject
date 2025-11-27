#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <omp.h>
#include "recon.h"
#include "metrics.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_video> <output_video>\n";
        return 1;
    }

    std::string inputPath  = argv[1];
    std::string outputPath = argv[2];
    int subsampleFactor    = (argc >= 4) ? std::stoi(argv[3]) : 2; // subsample window

    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open input video: " << inputPath << "\n";
        return 1;
    }

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    cv::VideoWriter writer(
        outputPath,
        cv::VideoWriter::fourcc('a', 'v', 'c', '1'), // tweak if needed
        fps,
        cv::Size(width, height)
    );

    if (!writer.isOpened()) {
        std::cerr << "Failed to open output video: " << outputPath << "\n";
        return 1;
    }

    cv::Mat frame;
    double totalMSE = 0.0;
    double totalSSIM = 0.0;
    int frameCount = 0;

    // State across frames
    cv::Mat prevRecon;       // R_{t-1}
    cv::Mat prevSubsampled;  // S_{t-1}

    const int    TILE_SIZE  = 2;
    const double SAD_THRESH = 1; // tune
    const int    ITERATIONS = 3;

    // Time comptuation start
    auto startTotal = std::chrono::high_resolution_clock::now();

    while (true) {
        if (!cap.read(frame)) break;  // end of video

        // 1) Frame subsampling
        cv::Mat subsampled, mask;
        subsampleFrame(frame, subsampled, mask, subsampleFactor);

        cv::Mat finalRecon;

        if (prevRecon.empty()) {
            // First frame: no temporal info, reconstruct whole frame
            reconstructNaive(subsampled, mask, subsampleFactor, finalRecon);
            iterativeRefine(finalRecon, mask, ITERATIONS);
        } else {
            // 2) Classify tiles using SAD on subsampled frames S_t and S_{t-1}
            cv::Mat tileActive;
            classifyTilesSADSubsample(subsampled, prevSubsampled,
                                      TILE_SIZE, SAD_THRESH,
                                      tileActive);

            // 3) Start from previous reconstruction: reuse static tiles by default
            finalRecon = prevRecon.clone();

            // 4) For active tiles, compute new reconstruction (bilinear) *just there*
            int tilesY = tileActive.rows;
            int tilesX = tileActive.cols;
            #pragma omp parallel
            {
                #pragma omp for schedule(static) nowait
                for (int ty = 0; ty < tilesY; ++ty) {
                    for (int tx = 0; tx < tilesX; ++tx) {
                        uint8_t active = tileActive.at<uint8_t>(ty, tx);
                        if (!active) continue; // static tile: keep prevRecon

                        int y0 = ty * TILE_SIZE;
                        int x0 = tx * TILE_SIZE;
                        int y1 = std::min(y0 + TILE_SIZE, height);
                        int x1 = std::min(x0 + TILE_SIZE, width);

                        for (int y = y0; y < y1; ++y) {
                            for (int x = x0; x < x1; ++x) {
                                if (mask.at<uint8_t>(y, x) == 1) {
                                    // known subsampled pixel
                                    finalRecon.at<cv::Vec3b>(y, x) =
                                        subsampled.at<cv::Vec3b>(y, x);
                                } else {
                                    // missing pixel: reconstruct using bilinear
                                    finalRecon.at<cv::Vec3b>(y, x) =
                                        bilinearPredict(subsampled, mask, y, x, subsampleFactor);
                                }
                            }
                        }
                    }
                }
            }
            // 5) Iterative refinement, but only do heavy smoothing on active tiles
            iterativeRefineTiles(finalRecon, mask, tileActive,
                                 TILE_SIZE, ITERATIONS);
        }

        // 6) Metrics on final reconstruction
        double mse = computeMSE(frame, finalRecon);
        double ssim = computeSSIM(frame, finalRecon);
        totalMSE += mse;
        totalSSIM += ssim;
        frameCount++;

        writer.write(finalRecon);

        // 7) Update state for next frame
        prevRecon      = finalRecon.clone();
        prevSubsampled = subsampled.clone();
    }

    if (frameCount > 0) {
        double avgMSE = totalMSE / frameCount;
        double avgSSIM = totalSSIM / frameCount;
        double psnr   = mseToPSNR(avgMSE);
        std::cout << "Avg MSE: " << avgMSE << "\n";
        std::cout << "Avg SSIM: " << avgSSIM << "\n";
        std::cout << "Avg PSNR: " << psnr << " dB\n";
    }

    std::cout << "Processed " << frameCount << " frames.\n";

    // Computation time print
    auto endTotal = std::chrono::high_resolution_clock::now();
    double totalSec =
        std::chrono::duration<double>(endTotal - startTotal).count();

    std::cout << "Total execution time: " << totalSec << " seconds\n";
    return 0;
}
