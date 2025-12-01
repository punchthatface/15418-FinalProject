// serial.cpp  (updated to use CUDA for active-tile reconstruction)

#include <iostream>
#include <opencv2/opencv.hpp>
#include "recon.h"
#include "metrics.h"
#include "recon_cuda.h"      // ← NEW include

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cout << "Usage: ./recon input.mp4 output.mp4 subsample_factor\n";
        return 1;
    }

    std::string inputPath  = argv[1];
    std::string outputPath = argv[2];
    int subsampleFactor    = atoi(argv[3]);

    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open input video\n";
        return 1;
    }

    int width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps    = (int)cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter writer(outputPath,
                           cv::VideoWriter::fourcc('m','p','4','v'),
                           fps,
                           cv::Size(width, height));

    cv::Mat frame, prevRecon, prevSubsampled;
    bool first = true;

    const int TILE_SIZE = 2;
    const int ITERATIONS = 4;
    const int SAD_THRESH = 10; // :(

    while (true) {
        if (!cap.read(frame))
            break;

        cv::Mat subsampled(height, width, CV_8UC3);
        cv::Mat mask(height, width, CV_8UC1);

        subsampleFrame(frame, subsampled, mask, subsampleFactor);

        cv::Mat finalRecon(height, width, CV_8UC3);

        if (first) {
            first = false;

            reconstructNaive(subsampled, mask,
                             subsampleFactor, finalRecon);

            iterativeRefine(finalRecon, mask, ITERATIONS);

        } else {
            cv::Mat tileActive;
            classifyTilesSADSubsample(subsampled, prevSubsampled,
                                      TILE_SIZE, SAD_THRESH,
                                      tileActive);

            // GPU replacement for big OpenMP tile loop
            reconstructActiveTilesCUDA(subsampled, mask,
                                       prevRecon, tileActive,
                                       TILE_SIZE, subsampleFactor,
                                       finalRecon);

            iterativeRefineTiles(finalRecon, mask, tileActive,
                                 TILE_SIZE, ITERATIONS);
        }

        double mse  = computeMSE(frame, finalRecon);
        double ssim = computeSSIM(frame, finalRecon);

        std::cout << "Frame MSE: " << mse
                  << "  SSIM: " << ssim << std::endl;

        writer.write(finalRecon);

        prevRecon      = finalRecon.clone();
        prevSubsampled = subsampled.clone();
    }

    std::cout << "Done.\n";
    return 0;
}
