#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <omp.h>

#include "recon.h"
#include "metrics.h"

#ifdef USE_CUDA_REFINEMENT
#include "recon_cuda.h"
#endif

using Clock = std::chrono::high_resolution_clock;

// Simple image-sequence reader: dir/frame_%06d.png or .jpg
struct ImageSequenceReader {
    std::string dir;
    int index;
    std::string ext; // ".png" or ".jpg"

    ImageSequenceReader(const std::string& d, const std::string& e = ".png")
        : dir(d), index(1), ext(e) {}

    void reset() { index = 1; }

    bool read(cv::Mat& out) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "%s/frame_%06d%s",
                      dir.c_str(), index++, ext.c_str());
        out = cv::imread(buf, cv::IMREAD_COLOR);
        return !out.empty();
    }
};

enum RunMode {
    MODE_CUDA,
    MODE_BASELINE
};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_video_or_frame_dir> <output_video_or_dash> [subsample_factor] [flags...]\n"
                  << "Flags:\n"
                  << "  --force-video    : force video input (VideoCapture only)\n"
                  << "  --force-images   : force image-sequence input (no VideoCapture)\n"
                  << "  --no-output      : do not write video output\n"
                  << "  --baseline       : CPU baseline (OpenMP forced to 1 thread, no CUDA)\n"
                  << "  --cuda           : CPU + CUDA refinement (default if no mode flag given)\n"
                  << "Notes:\n"
                  << "  - In image mode, expects <input>/frame_%06d.png or .jpg\n"
                  << "  - If <output> is '-' or 'none', output is disabled.\n";
        return 1;
    }

    std::string inputPath  = argv[1];
    std::string outputPath = argv[2];
    int subsampleFactor    = 2;

    if (argc >= 4 && argv[3][0] != '-') {
        subsampleFactor = std::stoi(argv[3]);
    }

    // Parse flags
    bool forceVideo    = false;
    bool forceImages   = false;
    bool disableOutput = false;

    RunMode mode = MODE_CUDA;  // default if no explicit mode flag

    for (int i = 3; i < argc; ++i) {
        std::string flag = argv[i];
        if (flag == "--force-video") {
            forceVideo = true;
        } else if (flag == "--force-images") {
            forceImages = true;
        } else if (flag == "--no-output") {
            disableOutput = true;
        } else if (flag == "--baseline") {
            mode = MODE_BASELINE;
        } else if (flag == "--cuda") {
            mode = MODE_CUDA;
        } else if (flag == "--omp-only") {
            std::cerr << "[Warn] --omp-only mode is no longer supported; treating as --baseline.\n";
            mode = MODE_BASELINE;
        }
    }

    // Check for conflicting input flags
    if (forceVideo && forceImages) {
        std::cerr << "Error: --force-video and --force-images cannot both be set.\n";
        return 1;
    }

    // If output path is "-" or "none", treat as no-output.
    if (outputPath == "-" || outputPath == "none") {
        disableOutput = true;
    }

    // Mode info
    if (mode == MODE_BASELINE) {
        std::cout << "[Init] Run mode: BASELINE (CPU, OpenMP forced to 1 thread, no CUDA)\n";
    } else {
        std::cout << "[Init] Run mode: CUDA (CPU + CUDA refinement)\n";
    }

    // Baseline: force OpenMP to 1 thread
    if (mode == MODE_BASELINE) {
        omp_set_dynamic(0);
        omp_set_num_threads(1);
    }

    // -------------------------------
    // Input mode selection
    // -------------------------------
    cv::VideoCapture cap;
    bool useVideoCapture = false;

    // Image reader: we'll try .png first; if first frame fails, we'll try .jpg
    ImageSequenceReader imgSeqPng(inputPath, ".png");
    ImageSequenceReader imgSeqJpg(inputPath, ".jpg");
    ImageSequenceReader* imgSeq = &imgSeqPng;

    cv::Mat firstFrame;
    int width  = 0;
    int height = 0;
    double fps = 30.0; // default

    if (forceImages) {
        std::cout << "[Init] Forcing image-sequence input from directory: "
                  << inputPath << "\n";

        // Try PNG first
        if (!imgSeqPng.read(firstFrame) || firstFrame.empty()) {
            // Try JPG
            imgSeq = &imgSeqJpg;
            imgSeqJpg.reset();
            if (!imgSeqJpg.read(firstFrame) || firstFrame.empty()) {
                std::cerr << "Failed to read first frame from directory (PNG or JPG): "
                          << inputPath << "/frame_000001.{png,jpg}\n";
                return 1;
            }
        } else {
            imgSeq = &imgSeqPng;
        }

        width  = firstFrame.cols;
        height = firstFrame.rows;
        fps    = 30.0; // arbitrary for VideoWriter

        imgSeq->reset(); // reset to start from frame_000001 in main loop
        useVideoCapture = false;
    } else if (forceVideo) {
        std::cout << "[Init] Forcing video input via VideoCapture: "
                  << inputPath << "\n";
        cap.open(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Error: failed to open video with --force-video: "
                      << inputPath << "\n";
            return 1;
        }
        if (!cap.read(firstFrame) || firstFrame.empty()) {
            std::cerr << "Error: unable to read first frame from video: "
                      << inputPath << "\n";
            return 1;
        }
        width  = firstFrame.cols;
        height = firstFrame.rows;
        fps    = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0.0 || std::isnan(fps)) {
            fps = 30.0;
        }
        // Rewind video
        cap.release();
        cap.open(inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Unexpected failure reopening video after probing.\n";
            return 1;
        }
        useVideoCapture = true;
    } else {
        // Auto mode: try VideoCapture, then fallback to images
        std::cout << "[Init] Auto input mode: trying VideoCapture first...\n";
        cap.open(inputPath);
        if (cap.isOpened()) {
            if (!cap.read(firstFrame) || firstFrame.empty()) {
                std::cerr << "Failed to read first frame from video: "
                          << inputPath << "\n";
                cap.release();
            } else {
                width  = firstFrame.cols;
                height = firstFrame.rows;
                fps    = cap.get(cv::CAP_PROP_FPS);
                if (fps <= 0.0 || std::isnan(fps)) {
                    fps = 30.0;
                }
                // Rewind
                cap.release();
                cap.open(inputPath);
                if (!cap.isOpened()) {
                    std::cerr << "Unexpected failure reopening video after probing.\n";
                    return 1;
                }
                useVideoCapture = true;
                std::cout << "[Init] Using VideoCapture for input: " << inputPath << "\n";
            }
        }

        if (!useVideoCapture) {
            std::cout << "[Init] VideoCapture failed; falling back to image-sequence mode from directory: "
                      << inputPath << "\n";

            // Try PNG first
            if (!imgSeqPng.read(firstFrame) || firstFrame.empty()) {
                // Try JPG
                imgSeq = &imgSeqJpg;
                imgSeqJpg.reset();
                if (!imgSeqJpg.read(firstFrame) || firstFrame.empty()) {
                    std::cerr << "Failed to read first frame from directory (PNG or JPG): "
                              << inputPath << "/frame_000001.{png,jpg}\n";
                    return 1;
                }
            } else {
                imgSeq = &imgSeqPng;
            }

            width  = firstFrame.cols;
            height = firstFrame.rows;
            fps    = 30.0; // arbitrary
            imgSeq->reset();
            useVideoCapture = false;
        }
    }

    // -------------------------------
    // Output video (optional)
    // -------------------------------
    bool writeOutput = !disableOutput;
    cv::VideoWriter writer;

    if (writeOutput) {
        writer.open(
            outputPath,
            cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
            fps,
            cv::Size(width, height)
        );

        if (!writer.isOpened()) {
            std::cerr << "Warning: failed to open output video: " << outputPath << "\n"
                      << "On PSC, this may be due to missing codecs. "
                      << "Continuing without video output.\n";
            writeOutput = false;
        } else {
            std::cout << "[Init] Output video: " << outputPath << "\n";
        }
    } else {
        std::cout << "[Init] Video output disabled.\n";
    }

    // -------------------------------
    // CUDA detection (once)
    // -------------------------------
#ifdef USE_CUDA_REFINEMENT
    bool cuda_ok = false;
    if (mode == MODE_CUDA) {
        cuda_ok = cudaRefinementInit();
        if (cuda_ok) {
            std::cout << "[Init] CUDA refinement enabled.\n";
        } else {
            std::cout << "[Init] CUDA NOT available — refinement will run on CPU.\n";
        }
    } else {
        std::cout << "[Init] CUDA refinement disabled in this mode.\n";
    }
#endif

    // -------------------------------
    // Timing accumulators
    // -------------------------------
    // Compute stages (we care most about these)
    double subsample_ms      = 0.0;
    double classify_ms       = 0.0;
    double recon_full_ms     = 0.0; // frame 0
    double recon_tiles_ms    = 0.0; // subsequent frames
    double refine_full_ms    = 0.0;
    double refine_tiles_ms   = 0.0;

    // I/O + metrics
    double input_io_ms       = 0.0; // imread / VideoCapture read
    double output_io_ms      = 0.0; // writer.write
    double metrics_ms        = 0.0;

    double totalMSE          = 0.0;
    double totalSSIM         = 0.0;
    int frameCount           = 0;

    // Tile statistics
    long long totalTiles       = 0;
    long long totalActiveTiles = 0;

    cv::Mat prevRecon;
    cv::Mat prevSubsampled;

    const int    TILE_SIZE   = 2;
    const double SAD_THRESH  = 1.0;
    const int    ITERATIONS  = 3;

    auto startTotal = Clock::now();

    // Abstracted frame reader
    auto readNextFrame = [&](cv::Mat& out)->bool {
        if (useVideoCapture) {
            return cap.read(out);
        } else {
            return imgSeq->read(out);
        }
    };

    cv::Mat frame;

    while (true) {
        // Measure input I/O time explicitly
        auto tRead0 = Clock::now();
        bool ok = readNextFrame(frame);
        auto tRead1 = Clock::now();
        input_io_ms += std::chrono::duration<double, std::milli>(tRead1 - tRead0).count();

        if (!ok || frame.empty()) break;

        // (1) Subsample
        cv::Mat subsampled, mask;
        {
            auto t0 = Clock::now();
            subsampleFrame(frame, subsampled, mask, subsampleFactor);
            auto t1 = Clock::now();
            subsample_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        cv::Mat finalRecon;

        if (prevRecon.empty()) {
            // FIRST FRAME (no temporal info)

            // Full-frame naive reconstruction
            {
                auto t0 = Clock::now();
                reconstructNaive(subsampled, mask, subsampleFactor, finalRecon);
                auto t1 = Clock::now();
                recon_full_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            }

            // Full-frame refinement
            {
                auto t0 = Clock::now();
#ifdef USE_CUDA_REFINEMENT
                if (mode == MODE_CUDA && cuda_ok) {
                    iterativeRefineCUDA(finalRecon, mask, ITERATIONS);
                } else {
                    iterativeRefine(finalRecon, mask, ITERATIONS);
                }
#else
                iterativeRefine(finalRecon, mask, ITERATIONS);
#endif
                auto t1 = Clock::now();
                refine_full_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            }

        } else {
            // SUBSEQUENT FRAMES

            // (2) Tile classification
            cv::Mat tileActive;
            {
                auto t0 = Clock::now();
                classifyTilesSADSubsample(subsampled, prevSubsampled,
                                          TILE_SIZE, SAD_THRESH,
                                          tileActive);
                auto t1 = Clock::now();
                classify_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            }

            finalRecon = prevRecon.clone();

            // Track tile statistics
            int tilesY = tileActive.rows;
            int tilesX = tileActive.cols;
            long long frameTiles = static_cast<long long>(tilesX) * tilesY;
            totalTiles += frameTiles;

            long long frameActive = 0;
            for (int ty = 0; ty < tilesY; ++ty) {
                for (int tx = 0; tx < tilesX; ++tx) {
                    if (tileActive.at<uint8_t>(ty, tx)) {
                        frameActive++;
                    }
                }
            }
            totalActiveTiles += frameActive;

            // (3) Reconstruction on active tiles
            {
                auto t0 = Clock::now();

                int rows   = subsampled.rows;
                int cols   = subsampled.cols;

                #pragma omp parallel
                {
                    #pragma omp for schedule(static) nowait
                    for (int ty = 0; ty < tilesY; ++ty) {
                        for (int tx = 0; tx < tilesX; ++tx) {
                            uint8_t active = tileActive.at<uint8_t>(ty, tx);
                            if (!active) continue;

                            int y0 = ty * TILE_SIZE;
                            int x0 = tx * TILE_SIZE;
                            int y1 = std::min(y0 + TILE_SIZE, rows);
                            int x1 = std::min(x0 + TILE_SIZE, cols);

                            for (int y = y0; y < y1; ++y) {
                                for (int x = x0; x < x1; ++x) {
                                    if (mask.at<uint8_t>(y, x) == 1) {
                                        finalRecon.at<cv::Vec3b>(y, x) =
                                            subsampled.at<cv::Vec3b>(y, x);
                                    } else {
                                        finalRecon.at<cv::Vec3b>(y, x) =
                                            bilinearPredict(subsampled, mask, y, x,
                                                            subsampleFactor);
                                    }
                                }
                            }
                        }
                    }
                }

                auto t1 = Clock::now();
                recon_tiles_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            }

            // (4) Refinement on active tiles
            {
                auto t0 = Clock::now();
#ifdef USE_CUDA_REFINEMENT
                if (mode == MODE_CUDA && cuda_ok) {
                    iterativeRefineTilesCUDA(finalRecon, mask, tileActive,
                                             TILE_SIZE, ITERATIONS);
                } else {
                    iterativeRefineTiles(finalRecon, mask, tileActive,
                                         TILE_SIZE, ITERATIONS);
                }
#else
                iterativeRefineTiles(finalRecon, mask, tileActive,
                                     TILE_SIZE, ITERATIONS);
#endif
                auto t1 = Clock::now();
                refine_tiles_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
        }

        // (5) Metrics (excluded from pipeline & compute timing)
        {
            auto t0 = Clock::now();
            double mse  = computeMSE(frame, finalRecon);
            double ssim = computeSSIM(frame, finalRecon);
            auto t1 = Clock::now();
            metrics_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

            totalMSE  += mse;
            totalSSIM += ssim;
        }

        frameCount++;

        // (6) Output (optional) – we still time it, but keep separate
        if (writeOutput) {
            auto t0 = Clock::now();
            writer.write(finalRecon);
            auto t1 = Clock::now();
            output_io_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        }

        prevRecon      = finalRecon.clone();
        prevSubsampled = subsampled.clone();
    }

    auto endTotal = Clock::now();
    double totalSec = std::chrono::duration<double>(endTotal - startTotal).count();
    double total_ms = totalSec * 1000.0;

    // Pipeline (no metrics)
    double pipeline_ms = total_ms - metrics_ms;
    if (pipeline_ms < 0) pipeline_ms = 0.0;
    double pipelineSec = pipeline_ms / 1000.0;

    // Compute = pipeline - input/output I/O
    double compute_ms = pipeline_ms - input_io_ms - output_io_ms;
    if (compute_ms < 0) compute_ms = 0.0;
    double computeSec = compute_ms / 1000.0;

    double avgMSE  = (frameCount > 0) ? (totalMSE  / frameCount) : 0.0;
    double avgSSIM = (frameCount > 0) ? (totalSSIM / frameCount) : 0.0;
    double psnr    = mseToPSNR(avgMSE);

    // ---------------------------------------
    // High-level summary
    // ---------------------------------------
    std::cout << "\n================= RESULTS =================\n";
    std::cout << "Processed frames: " << frameCount << "\n";
    std::cout << "Subsample factor: " << subsampleFactor << "\n";

#ifdef USE_CUDA_REFINEMENT
    if (mode == MODE_BASELINE) {
        std::cout << "Refinement backend: CPU baseline (1 thread, no CUDA)\n";
    } else {
        std::cout << "Refinement backend: "
                  << (cuda_ok ? "CUDA\n" : "CPU (CUDA requested, fallback)\n");
    }
#else
    std::cout << "Refinement backend: CPU (OpenMP)\n";
#endif

    std::cout << "Avg MSE:   " << avgMSE  << "\n";
    std::cout << "Avg SSIM:  " << avgSSIM << "\n";
    std::cout << "Avg PSNR:  " << psnr    << " dB\n\n";

    std::cout << "Total time (ALL, incl. metrics & I/O): "
              << totalSec << " sec\n";
    std::cout << "Pipeline (no metrics, but incl. I/O): "
              << pipelineSec << " sec total, ";
    if (frameCount > 0) {
        std::cout << (pipeline_ms / frameCount) << " ms/frame\n";
    } else {
        std::cout << "N/A ms/frame\n";
    }

    std::cout << "  - Input I/O (read frames): "
              << (input_io_ms / 1000.0) << " sec total, ";
    if (frameCount > 0) {
        std::cout << (input_io_ms / frameCount) << " ms/frame\n";
    } else {
        std::cout << "N/A ms/frame\n";
    }

    std::cout << "  - Output I/O (write frames): "
              << (output_io_ms / 1000.0) << " sec total, ";
    if (frameCount > 0) {
        std::cout << (output_io_ms / frameCount) << " ms/frame\n";
    } else {
        std::cout << "N/A ms/frame\n";
    }

    std::cout << "Compute-only (no metrics, no I/O): "
              << computeSec << " sec total, ";
    if (frameCount > 0) {
        std::cout << (compute_ms / frameCount) << " ms/frame\n\n";
    } else {
        std::cout << "N/A ms/frame\n\n";
    }

    std::cout << "Metrics time: "
              << (metrics_ms / 1000.0) << " sec total, ";
    if (frameCount > 0) {
        std::cout << (metrics_ms / frameCount) << " ms/frame\n\n";
    } else {
        std::cout << "N/A ms/frame\n\n";
    }

    // ---------------------------------------
    // Compute-stage breakdown (no metrics, no I/O)
    // ---------------------------------------
    auto print_compute_stage = [&](const char* name, double ms){
        double avg = (frameCount > 0) ? (ms / frameCount) : 0.0;
        double pct = (compute_ms > 0.0 ? (100.0 * ms / compute_ms) : 0.0);
        std::cout << name << ": "
                  << ms << " ms total, "
                  << avg << " ms/frame, "
                  << pct << "% of compute\n";
    };

    std::cout << "====== Compute breakdown (no metrics, no I/O) ======\n";
    print_compute_stage("Subsample               ", subsample_ms);
    print_compute_stage("Tile classify           ", classify_ms);
    print_compute_stage("Reconstruction (full)   ", recon_full_ms);
    print_compute_stage("Reconstruction (tiles)  ", recon_tiles_ms);
    print_compute_stage("Refinement (full)       ", refine_full_ms);
    print_compute_stage("Refinement (tiles)      ", refine_tiles_ms);
    std::cout << "Total compute            : "
              << compute_ms << " ms\n";
    std::cout << "====================================================\n\n";

    // ---------------------------------------
    // Tile statistics
    // ---------------------------------------
    if (totalTiles > 0 && frameCount > 1) {
        double avgActiveFrac  = static_cast<double>(totalActiveTiles) /
                                static_cast<double>(totalTiles);
        double avgActiveTiles = static_cast<double>(totalActiveTiles) /
                                static_cast<double>(frameCount - 1); // frame 0 has no tiles

        std::cout << "[Tile stats] total tiles (frames>0): " << totalTiles << "\n";
        std::cout << "[Tile stats] total active tiles     : " << totalActiveTiles << "\n";
        std::cout << "[Tile stats] avg active fraction    : "
                  << (avgActiveFrac * 100.0) << "%\n";
        std::cout << "[Tile stats] avg active tiles/frame : "
                  << avgActiveTiles << " (for frames 1..N-1)\n\n";
    }

#ifdef USE_CUDA_REFINEMENT
    cudaPrintRefineStats();
#endif

    return 0;
}
