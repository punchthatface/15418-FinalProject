#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <chrono>

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

    // Time comptuation start
    auto startTotal = std::chrono::high_resolution_clock::now();

    while (true) {
        if (!cap.read(frame)) break;  // end of video


        // 1) Frame subsampling
        cv::Mat subsampled, mask;
        subsampleFrame(frame, subsampled, mask, subsampleFactor);
        
        // 2) Spatial prediction
        cv::Mat spatialRecon;
        reconstructNaive(subsampled, mask, subsampleFactor, spatialRecon);

        // 3) Iterative local refinement
        int iterations = 3;  // tune: 2-5 typical
        iterativeRefine(spatialRecon, mask, iterations);ew


        double mse = computeMSE(frame, spatialRecon);
        double ssim = computeSSIM(frame, spatialRecon);
        totalMSE += mse;
        totalSSIM += ssim;
        frameCount++;

        writer.write(spatialRecon);
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
