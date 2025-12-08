#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <sys/types.h>

#include <opencv2/opencv.hpp>
#include <omp.h>

#include "recon.h"
#include "metrics.h"

#ifdef USE_CUDA_REFINEMENT
#include "recon_cuda.h"
#endif

using Clock = std::chrono::high_resolution_clock;

enum class RunMode {
    BASELINE,   // OpenMP forced to 1 thread, no CUDA
    OMP_ONLY,   // OpenMP multi-thread, no CUDA
    CUDA        // CPU + CUDA refinement / reconstruction
};

struct Args {
    std::string inputPath;
    std::string outputPath;
    int subsampleFactor = 2;

    bool forceImages     = false;
    bool outputEnabled   = true;
    bool metricsEnabled  = true;

    RunMode mode         = RunMode::CUDA;
    bool cudaClassify    = false;   // NEW: use CUDA for tile classification
};

// ----------------- Arg parsing -----------------

static Args parseArgs(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_video | image_dir> <output_video | -> [subsample_factor] [flags]\n"
                  << "Flags:\n"
                  << "  --force-images    : treat input as image-sequence directory\n"
                  << "  --no-output       : do not write output video\n"
                  << "  --baseline        : baseline mode (1 thread, no CUDA)\n"
                  << "  --omp-only        : OpenMP-only mode (multi-thread, no CUDA)\n"
                  << "  (default, if CUDA compiled) : CUDA mode (CPU + CUDA refinement)\n"
                  << "  --no-metrics      : disable MSE/SSIM/PSNR computation\n"
                  << "  --cuda-classify   : EXPERIMENTAL: run classifyTilesSADSubsample on CUDA\n";
        std::exit(1);
    }

    Args args;
    args.inputPath  = argv[1];
    args.outputPath = argv[2];

    // Optional subsample factor if given and not a flag
    if (argc >= 4 && argv[3][0] != '-') {
        args.subsampleFactor = std::max(1, std::atoi(argv[3]));
    }

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--force-images") {
            args.forceImages = true;
        } else if (a == "--no-output") {
            args.outputEnabled = false;
        } else if (a == "--baseline") {
            args.mode = RunMode::BASELINE;
        } else if (a == "--omp-only") {
            args.mode = RunMode::OMP_ONLY;
        } else if (a == "--no-metrics") {
            args.metricsEnabled = false;
        } else if (a == "--cuda-classify") {
            args.cudaClassify = true;
        }
    }

    if (args.outputPath == "-") {
        args.outputEnabled = false;
    }

#ifndef USE_CUDA_REFINEMENT
    // If we compiled without CUDA, force to OMP-only and ignore cudaClassify flag.
    if (args.mode == RunMode::CUDA) {
        args.mode = RunMode::OMP_ONLY;
    }
    args.cudaClassify = false;
#endif

    return args;
}

// ----------------- Image-sequence helpers -----------------

static bool hasImageExtension(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower.size() < 4) return false;
    if (lower.rfind(".png")  == lower.size() - 4) return true;
    if (lower.rfind(".jpg")  == lower.size() - 4) return true;
    if (lower.rfind(".jpeg") == lower.size() - 5) return true;
    return false;
}

static std::vector<std::string> listImagesInDir(const std::string& dir) {
    std::vector<std::string> files;
    DIR* d = opendir(dir.c_str());
    if (!d) {
        std::perror("opendir");
        return files;
    }
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name = ent->d_name;
        if (name == "." || name == "..") continue;
        if (hasImageExtension(name)) {
            files.push_back(name);
        }
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    return files;
}

// ----------------- main -----------------

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    const int    TILE_SIZE   = 2;
    const double SAD_THRESH  = 1.0;
    const int    ITERATIONS  = 3;

    // Configure OpenMP threads according to mode
    if (args.mode == RunMode::BASELINE) {
        omp_set_num_threads(1);
        std::cout << "[Init] Run mode: BASELINE (CPU, OpenMP forced to 1 thread, no CUDA)\n";
    } else if (args.mode == RunMode::OMP_ONLY) {
        std::cout << "[Init] Run mode: OMP-ONLY (CPU, OpenMP multi-thread, no CUDA)\n";
    } else {
        std::cout << "[Init] Run mode: CUDA (CPU + CUDA refinement)\n";
    }

#ifdef USE_CUDA_REFINEMENT
    if (args.mode == RunMode::CUDA) {
        if (!cudaRefinementInit()) {
            std::cout << "[Init] CUDA not available, falling back to OMP-ONLY.\n";
            args.mode = RunMode::OMP_ONLY;
            args.cudaClassify = false;
        } else if (args.cudaClassify) {
            std::cout << "[Init] EXPERIMENT: CUDA classification enabled.\n";
        }
    } else {
        args.cudaClassify = false;
    }
#else
    if (args.cudaClassify) {
        std::cout << "[Init] --cuda-classify ignored (no CUDA support in this build).\n";
        args.cudaClassify = false;
    }
#endif

    // Input setup: video vs image-sequence
    bool useImageSeq = args.forceImages;
    cv::VideoCapture cap;
    std::vector<std::string> imageFiles;

    if (!useImageSeq) {
        cap.open(args.inputPath);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open video input: " << args.inputPath << "\n";
            std::cerr << "Hint: use --force-images if this is a directory of frames.\n";
            return 1;
        }
    } else {
        std::cout << "[Init] Forcing image-sequence input from directory: "
                  << args.inputPath << "\n";
        imageFiles = listImagesInDir(args.inputPath);
        if (imageFiles.empty()) {
            std::cerr << "No images found in directory: " << args.inputPath << "\n";
            return 1;
        }
    }

    if (!args.outputEnabled) {
        std::cout << "[Init] Video output disabled.\n";
    }
    if (!args.metricsEnabled) {
        std::cout << "[Init] Metrics disabled (no MSE/SSIM/PSNR).\n";
    }

    // Determine frame size & fps
    cv::Size frameSize;
    double fps = 30.0;

    if (!useImageSeq) {
        frameSize = cv::Size(
            (int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
            (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)
        );
        fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0.0 || std::isnan(fps)) {
            fps = 30.0;
        }
    } else {
        std::string firstPath = args.inputPath + "/" + imageFiles[0];
        cv::Mat tmp = cv::imread(firstPath, cv::IMREAD_COLOR);
        if (tmp.empty()) {
            std::cerr << "Failed to read first image: " << firstPath << "\n";
            return 1;
        }
        frameSize = tmp.size();
        fps = 30.0;
    }

    // Output writer
    cv::VideoWriter writer;
    if (args.outputEnabled) {
        writer.open(args.outputPath,
                    cv::VideoWriter::fourcc('m','p','4','v'),
                    fps,
                    frameSize);
        if (!writer.isOpened()) {
            std::cerr << "Failed to open output video: " << args.outputPath << "\n";
            return 1;
        }
    }

    // Timing accumulators (ms)
    double total_input_ms   = 0.0;
    double total_output_ms  = 0.0;
    double total_compute_ms = 0.0;
    double total_metrics_ms = 0.0;

    double subsample_ms    = 0.0;
    double classify_ms     = 0.0;
    double recon_full_ms   = 0.0;
    double recon_tiles_ms  = 0.0;
    double refine_full_ms  = 0.0;
    double refine_tiles_ms = 0.0;

    // Tile statistics
    long long totalTiles_sum        = 0;
    long long totalActiveTiles_sum  = 0;
    int       framesWithTiles       = 0;

    // Metrics accumulators
    double sumMSE   = 0.0;
    double sumSSIM  = 0.0;
    int    metricFrames = 0;

    auto t_all_start = Clock::now();

    cv::Mat prevRecon;
    cv::Mat prevSubsampled;

    int frameIndex      = 0;
    int processedFrames = 0;

    while (true) {
        cv::Mat frame;

        // Input I/O timing
        auto t_in_start = Clock::now();
        if (!useImageSeq) {
            if (!cap.read(frame) || frame.empty()) {
                break;
            }
        } else {
            if (frameIndex >= (int)imageFiles.size()) {
                break;
            }
            std::string path = args.inputPath + "/" + imageFiles[frameIndex];
            frame = cv::imread(path, cv::IMREAD_COLOR);
            if (frame.empty()) {
                std::cerr << "Failed to read image: " << path << " (skipping)\n";
                frameIndex++;
                continue;
            }
        }
        auto t_in_end = Clock::now();
        total_input_ms += std::chrono::duration<double, std::milli>(t_in_end - t_in_start).count();

        if (frame.size() != frameSize) {
            cv::resize(frame, frame, frameSize);
        }

        auto t_compute_start = Clock::now();

        // Subsample
        cv::Mat subsampled, mask;
        auto t_sub_start = Clock::now();
        subsampleFrame(frame, subsampled, mask, args.subsampleFactor);
        auto t_sub_end = Clock::now();
        subsample_ms += std::chrono::duration<double, std::milli>(t_sub_end - t_sub_start).count();

        cv::Mat finalRecon;

        if (frameIndex == 0) {
            // First frame: full reconstruction + full refinement
            auto t_recon_full_start = Clock::now();
            reconstructNaive(subsampled, mask, args.subsampleFactor, finalRecon);
            auto t_recon_full_end = Clock::now();
            recon_full_ms += std::chrono::duration<double, std::milli>(
                                 t_recon_full_end - t_recon_full_start).count();

            auto t_refine_full_start = Clock::now();
            if (args.mode == RunMode::CUDA) {
#ifdef USE_CUDA_REFINEMENT
                iterativeRefineCUDA(finalRecon, mask, ITERATIONS);
#else
                iterativeRefine(finalRecon, mask, ITERATIONS);
#endif
            } else {
                iterativeRefine(finalRecon, mask, ITERATIONS);
            }
            auto t_refine_full_end = Clock::now();
            refine_full_ms += std::chrono::duration<double, std::milli>(
                                  t_refine_full_end - t_refine_full_start).count();
        } else {
            // Subsequent frames: tile classify, tile reconstruction, tile refinement
            cv::Mat tileActive;

            auto t_classify_start = Clock::now();
            if (args.mode == RunMode::CUDA && args.cudaClassify) {
#ifdef USE_CUDA_REFINEMENT
                classifyTilesSADSubsampleCUDA(subsampled, prevSubsampled,
                                              TILE_SIZE, SAD_THRESH,
                                              tileActive);
#else
                classifyTilesSADSubsample(subsampled, prevSubsampled,
                                          TILE_SIZE, SAD_THRESH,
                                          tileActive);
#endif
            } else {
                classifyTilesSADSubsample(subsampled, prevSubsampled,
                                          TILE_SIZE, SAD_THRESH,
                                          tileActive);
            }
            auto t_classify_end = Clock::now();
            classify_ms += std::chrono::duration<double, std::milli>(
                               t_classify_end - t_classify_start).count();

            int tilesY = tileActive.rows;
            int tilesX = tileActive.cols;
            long long tilesThisFrame  = (long long)tilesY * tilesX;
            long long activeThisFrame = 0;
            for (int ty = 0; ty < tilesY; ++ty) {
                const uint8_t* rowPtr = tileActive.ptr<uint8_t>(ty);
                for (int tx = 0; tx < tilesX; ++tx) {
                    if (rowPtr[tx] != 0) activeThisFrame++;
                }
            }
            totalTiles_sum       += tilesThisFrame;
            totalActiveTiles_sum += activeThisFrame;
            framesWithTiles++;

            // Tile reconstruction
            auto t_recon_tiles_start = Clock::now();
            if (args.mode == RunMode::CUDA) {
#ifdef USE_CUDA_REFINEMENT
                reconstructTilesCUDA(finalRecon,
                                     prevRecon,
                                     subsampled,
                                     mask,
                                     tileActive,
                                     TILE_SIZE,
                                     args.subsampleFactor);
#else
                finalRecon = prevRecon.clone();
                int height = frame.rows;
                int width  = frame.cols;
                #pragma omp parallel for schedule(static)
                for (int ty = 0; ty < tilesY; ++ty) {
                    for (int tx = 0; tx < tilesX; ++tx) {
                        uint8_t active = tileActive.at<uint8_t>(ty, tx);
                        if (!active) continue;
                        int y0 = ty * TILE_SIZE;
                        int x0 = tx * TILE_SIZE;
                        int y1 = std::min(y0 + TILE_SIZE, height);
                        int x1 = std::min(x0 + TILE_SIZE, width);
                        for (int y = y0; y < y1; ++y) {
                            for (int x = x0; x < x1; ++x) {
                                if (mask.at<uint8_t>(y, x) == 1) {
                                    finalRecon.at<cv::Vec3b>(y, x) =
                                        subsampled.at<cv::Vec3b>(y, x);
                                } else {
                                    finalRecon.at<cv::Vec3b>(y, x) =
                                        bilinearPredict(subsampled, mask, y, x, args.subsampleFactor);
                                }
                            }
                        }
                    }
                }
#endif
            } else {
                finalRecon = prevRecon.clone();
                int height = frame.rows;
                int width  = frame.cols;
                #pragma omp parallel for schedule(static)
                for (int ty = 0; ty < tilesY; ++ty) {
                    for (int tx = 0; tx < tilesX; ++tx) {
                        uint8_t active = tileActive.at<uint8_t>(ty, tx);
                        if (!active) continue;
                        int y0 = ty * TILE_SIZE;
                        int x0 = tx * TILE_SIZE;
                        int y1 = std::min(y0 + TILE_SIZE, height);
                        int x1 = std::min(x0 + TILE_SIZE, width);
                        for (int y = y0; y < y1; ++y) {
                            for (int x = x0; x < x1; ++x) {
                                if (mask.at<uint8_t>(y, x) == 1) {
                                    finalRecon.at<cv::Vec3b>(y, x) =
                                        subsampled.at<cv::Vec3b>(y, x);
                                } else {
                                    finalRecon.at<cv::Vec3b>(y, x) =
                                        bilinearPredict(subsampled, mask, y, x, args.subsampleFactor);
                                }
                            }
                        }
                    }
                }
            }
            auto t_recon_tiles_end = Clock::now();
            recon_tiles_ms += std::chrono::duration<double, std::milli>(
                                  t_recon_tiles_end - t_recon_tiles_start).count();

            // Tile refinement
            auto t_refine_tiles_start = Clock::now();
            if (args.mode == RunMode::CUDA) {
#ifdef USE_CUDA_REFINEMENT
                iterativeRefineTilesCUDA(finalRecon, mask, tileActive, TILE_SIZE, ITERATIONS);
#else
                iterativeRefineTiles(finalRecon, mask, tileActive, TILE_SIZE, ITERATIONS);
#endif
            } else {
                iterativeRefineTiles(finalRecon, mask, tileActive, TILE_SIZE, ITERATIONS);
            }
            auto t_refine_tiles_end = Clock::now();
            refine_tiles_ms += std::chrono::duration<double, std::milli>(
                                   t_refine_tiles_end - t_refine_tiles_start).count();
        }

        auto t_compute_end = Clock::now();
        total_compute_ms += std::chrono::duration<double, std::milli>(
                                t_compute_end - t_compute_start).count();

        // Output
        auto t_out_start = Clock::now();
        if (args.outputEnabled) {
            writer.write(finalRecon);
        }
        auto t_out_end = Clock::now();
        total_output_ms += std::chrono::duration<double, std::milli>(
                               t_out_end - t_out_start).count();

        // Metrics
        if (args.metricsEnabled) {
            auto t_metrics_start = Clock::now();
            double mse  = computeMSE(frame, finalRecon);
            double ssim = computeSSIM(frame, finalRecon);
            sumMSE  += mse;
            sumSSIM += ssim;
            metricFrames++;
            auto t_metrics_end = Clock::now();
            total_metrics_ms += std::chrono::duration<double, std::milli>(
                                    t_metrics_end - t_metrics_start).count();
        }

        prevRecon       = finalRecon.clone();
        prevSubsampled  = subsampled.clone();

        frameIndex++;
        processedFrames++;
    }

    auto t_all_end = Clock::now();
    double total_all_sec = std::chrono::duration<double>(
                               t_all_end - t_all_start).count();

    double pipeline_ms = total_input_ms + total_output_ms + total_compute_ms;
    double compute_ms  = total_compute_ms;
    double metrics_ms  = total_metrics_ms;

    std::cout << "\n================= RESULTS =================\n";
    std::cout << "Processed frames: " << processedFrames << "\n";
    std::cout << "Subsample factor: " << args.subsampleFactor << "\n";

    std::cout << "Refinement backend: ";
    if (args.mode == RunMode::CUDA) {
        std::cout << "CUDA";
        if (args.cudaClassify) std::cout << " (with CUDA classify experiment)";
        std::cout << "\n";
    } else if (args.mode == RunMode::OMP_ONLY) {
        std::cout << "CPU (OpenMP, no CUDA)\n";
    } else {
        std::cout << "CPU baseline (1 thread, no CUDA)\n";
    }

    if (args.metricsEnabled && metricFrames > 0) {
        double avgMSE  = sumMSE  / metricFrames;
        double avgSSIM = sumSSIM / metricFrames;
        double psnr    = mseToPSNR(avgMSE);
        std::cout << "Avg MSE:   " << avgMSE  << "\n";
        std::cout << "Avg SSIM:  " << avgSSIM << "\n";
        std::cout << "Avg PSNR:  " << psnr    << " dB\n";
    } else {
        std::cout << "Avg MSE:   N/A (metrics disabled)\n";
        std::cout << "Avg SSIM:  N/A (metrics disabled)\n";
        std::cout << "Avg PSNR:  N/A (metrics disabled)\n";
    }

    double total_pipeline_sec = pipeline_ms / 1000.0;
    double total_compute_sec  = compute_ms  / 1000.0;
    double total_metrics_sec  = metrics_ms  / 1000.0;

    std::cout << "\nTotal time (ALL, incl. metrics & I/O): "
              << total_all_sec << " sec\n";
    std::cout << "Pipeline (no metrics, but incl. I/O): "
              << total_pipeline_sec << " sec total, ";
    if (processedFrames > 0) {
        std::cout << (pipeline_ms / processedFrames) << " ms/frame\n";
    } else {
        std::cout << "N/A ms/frame\n";
    }

    std::cout << "  - Input I/O (read frames): "
              << (total_input_ms / 1000.0) << " sec total, ";
    if (processedFrames > 0) {
        std::cout << (total_input_ms / processedFrames) << " ms/frame\n";
    } else {
        std::cout << "N/A ms/frame\n";
    }

    std::cout << "  - Output I/O (write frames): "
              << (total_output_ms / 1000.0) << " sec total, ";
    if (processedFrames > 0) {
        std::cout << (total_output_ms / processedFrames) << " ms/frame\n";
    } else {
        std::cout << "N/A ms/frame\n";
    }

    std::cout << "Compute-only (no metrics, no I/O): "
              << total_compute_sec << " sec total, ";
    if (processedFrames > 0) {
        std::cout << (compute_ms / processedFrames) << " ms/frame\n";
    } else {
        std::cout << "N/A ms/frame\n";
    }
    std::cout << "\nMetrics time: "
              << total_metrics_sec << " sec";
    if (processedFrames > 0) {
        std::cout << " (" << (metrics_ms / processedFrames) << " ms/frame)";
    }
    if (!args.metricsEnabled) {
        std::cout << " (metrics disabled)";
    }
    std::cout << "\n\n";

    std::cout << "====== Compute breakdown (no metrics, no I/O) ======\n";
    if (processedFrames > 0 && compute_ms > 0.0) {
        auto pct = [&](double ms) {
            return (ms / compute_ms) * 100.0;
        };
        auto perFrame = [&](double ms) {
            return ms / processedFrames;
        };

        std::cout << "Subsample               : " << subsample_ms / 1000.0
                  << " ms total, " << perFrame(subsample_ms)
                  << " ms/frame, " << pct(subsample_ms) << "% of compute\n";
        std::cout << "Tile classify           : " << classify_ms / 1000.0
                  << " ms total, " << perFrame(classify_ms)
                  << " ms/frame, " << pct(classify_ms) << "% of compute\n";
        std::cout << "Reconstruction (full)   : " << recon_full_ms / 1000.0
                  << " ms total, " << perFrame(recon_full_ms)
                  << " ms/frame, " << pct(recon_full_ms) << "% of compute\n";
        std::cout << "Reconstruction (tiles)  : " << recon_tiles_ms / 1000.0
                  << " ms total, " << perFrame(recon_tiles_ms)
                  << " ms/frame, " << pct(recon_tiles_ms) << "% of compute\n";
        std::cout << "Refinement (full)       : " << refine_full_ms / 1000.0
                  << " ms total, " << perFrame(refine_full_ms)
                  << " ms/frame, " << pct(refine_full_ms) << "% of compute\n";
        std::cout << "Refinement (tiles)      : " << refine_tiles_ms / 1000.0
                  << " ms total, " << perFrame(refine_tiles_ms)
                  << " ms/frame, " << pct(refine_tiles_ms) << "% of compute\n";
        std::cout << "Total compute            : " << compute_ms / 1000.0 << " ms\n";
    } else {
        std::cout << "Not enough data to compute breakdown.\n";
    }
    std::cout << "====================================================\n\n";

    if (framesWithTiles > 0) {
        double avgTilesPerFrame       = (double)totalTiles_sum / framesWithTiles;
        double avgActiveTilesPerFrame = (double)totalActiveTiles_sum / framesWithTiles;
        double avgActiveFraction      = (avgTilesPerFrame > 0.0)
            ? (avgActiveTilesPerFrame / avgTilesPerFrame) * 100.0
            : 0.0;

        std::cout << "[Tile stats] total tiles (frames>0): "
                  << totalTiles_sum << "\n";
        std::cout << "[Tile stats] total active tiles     : "
                  << totalActiveTiles_sum << "\n";
        std::cout << "[Tile stats] avg active fraction    : "
                  << avgActiveFraction << "%\n";
        std::cout << "[Tile stats] avg active tiles/frame : "
                  << avgActiveTilesPerFrame
                  << " (for frames 1..N-1)\n\n";
    }

#ifdef USE_CUDA_REFINEMENT
    cudaPrintRefineStats();
#else
    std::cout << "[CUDA refine stats]\n";
    std::cout << "  device          : none/unknown\n";
    std::cout << "  refine calls    : 0 (full-frame)\n";
    std::cout << "  refineTiles calls: 0 (tile-aware)\n";
    std::cout << "  reconTiles calls: 0 (tile-aware)\n";
    std::cout << "  total cudaMalloc: 0 ms\n";
    std::cout << "  total cudaFree  : 0 ms\n";
    std::cout << "  total H2D       : 0 ms\n";
    std::cout << "  total kernel    : 0 ms\n";
    std::cout << "  total D2H       : 0 ms\n";
    std::cout << "--------------------------------------------------\n";
#endif

    return 0;
}
