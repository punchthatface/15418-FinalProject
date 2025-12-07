// recon_cuda.cu
#include "recon_cuda.h"
#include "recon.h"        // for CPU fallback (iterativeRefine*, bilinearPredict if needed)

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <cstdio>

using Clock = std::chrono::high_resolution_clock;

// -------------------- Global CUDA state --------------------

static bool   g_cuda_initialized = false;
static bool   g_cuda_ok          = false;
static int    g_cuda_device_id   = -1;
static char   g_cuda_device_name[256] = "none/unknown";

// -------------------- Profiling accumulators (ms) --------------------

static double g_malloc_ms      = 0.0;
static double g_free_ms        = 0.0;
static double g_h2d_ms         = 0.0;
static double g_kernel_ms      = 0.0;
static double g_d2h_ms         = 0.0;

static int    g_refine_calls       = 0;
static int    g_refine_tiles_calls = 0;
static int    g_recon_tiles_calls  = 0;  // new: for reconstruction on tiles

// -------------------- Persistent device buffers --------------------
// Reused across frames to avoid per-call cudaMalloc/cudaFree.

static uchar3*        g_d_img_cur        = nullptr; // recon / temp
static uchar3*        g_d_img_next       = nullptr; // recon / temp
static uchar3*        g_d_subsampled     = nullptr; // subsampled frame
static unsigned char* g_d_mask_dev       = nullptr; // mask for known pixels
static unsigned char* g_d_tileActive_dev = nullptr; // active tiles

static size_t g_capacity_pixels = 0;  // pixels capacity
static size_t g_capacity_tiles  = 0;  // tiles capacity

// -------------------- Error helper --------------------

static inline void checkCuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Error at " << what << ": "
                  << cudaGetErrorString(err) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

// -------------------- Init --------------------

bool cudaRefinementInit() {
    if (g_cuda_initialized) {
        return g_cuda_ok;
    }
    g_cuda_initialized = true;

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        std::cerr << "[CUDA] No CUDA-capable device detected.\n";
        g_cuda_ok = false;
        return false;
    }

    g_cuda_device_id = 0;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, g_cuda_device_id) == cudaSuccess) {
        std::snprintf(
            g_cuda_device_name,
            sizeof(g_cuda_device_name),
            "%s (cc %d.%d, %d SMs)",
            prop.name,
            prop.major,
            prop.minor,
            prop.multiProcessorCount
        );
    } else {
        std::snprintf(g_cuda_device_name, sizeof(g_cuda_device_name),
                      "device_%d (unknown props)", g_cuda_device_id);
    }

    if (cudaSetDevice(g_cuda_device_id) != cudaSuccess) {
        std::cerr << "[CUDA] Failed to set device " << g_cuda_device_id << "\n";
        g_cuda_ok = false;
        return false;
    }

    std::cerr << "[CUDA] Using device " << g_cuda_device_id
              << ": " << g_cuda_device_name << "\n";

    g_malloc_ms = g_free_ms = g_h2d_ms = g_kernel_ms = g_d2h_ms = 0.0;
    g_refine_calls = g_refine_tiles_calls = g_recon_tiles_calls = 0;
    g_capacity_pixels = 0;
    g_capacity_tiles  = 0;

    g_cuda_ok = true;
    return true;
}

// -------------------- Device helpers --------------------

__device__ inline int idx2d(int y, int x, int cols) {
    return y * cols + x;
}

// -------------------- Kernels: refinement (existing) --------------------

// Small 3x3 refinement kernel (full frame)
__global__
void kernelIterativeRefineFull(const uchar3* __restrict__ cur,
                               uchar3* __restrict__       next,
                               const unsigned char* __restrict__ mask,
                               int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int idx = idx2d(y, x, cols);

    if (mask[idx] == 1) {
        next[idx] = cur[idx];
        return;
    }

    int sumB = 0, sumG = 0, sumR = 0;
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        int ny = y + dy;
        if (ny < 0 || ny >= rows) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            if (nx < 0 || nx >= cols) continue;
            int nidx = idx2d(ny, nx, cols);
            uchar3 c = cur[nidx];
            sumB += c.x;
            sumG += c.y;
            sumR += c.z;
            count++;
        }
    }

    if (count == 0) {
        next[idx] = cur[idx];
    } else {
        next[idx] = make_uchar3(
            static_cast<unsigned char>(sumB / count),
            static_cast<unsigned char>(sumG / count),
            static_cast<unsigned char>(sumR / count)
        );
    }
}

// Tile-aware refinement kernel
__global__
void kernelIterativeRefineTiles(const uchar3* __restrict__ cur,
                                uchar3* __restrict__       next,
                                const unsigned char* __restrict__ mask,
                                const unsigned char* __restrict__ tileActive,
                                int rows, int cols,
                                int tilesY, int tilesX,
                                int tileSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int idx = idx2d(y, x, cols);

    int ty = y / tileSize;
    int tx = x / tileSize;
    if (ty < 0 || ty >= tilesY || tx < 0 || tx >= tilesX) {
        next[idx] = cur[idx];
        return;
    }

    int tileIdx = ty * tilesX + tx;
    unsigned char active = tileActive[tileIdx];

    if (!active) {
        next[idx] = cur[idx];
        return;
    }

    if (mask[idx] == 1) {
        next[idx] = cur[idx];
        return;
    }

    int sumB = 0, sumG = 0, sumR = 0;
    int count = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        int ny = y + dy;
        if (ny < 0 || ny >= rows) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            if (nx < 0 || nx >= cols) continue;
            int nidx = idx2d(ny, nx, cols);
            uchar3 c = cur[nidx];
            sumB += c.x;
            sumG += c.y;
            sumR += c.z;
            count++;
        }
    }

    if (count == 0) {
        next[idx] = cur[idx];
    } else {
        next[idx] = make_uchar3(
            static_cast<unsigned char>(sumB / count),
            static_cast<unsigned char>(sumG / count),
            static_cast<unsigned char>(sumR / count)
        );
    }
}

// -------------------- Kernel: reconstruction on tiles (new) --------------------

// Mirrors CPU logic:
// - base image = prevRecon
// - if tile inactive -> keep prevRecon
// - if tile active:
//     if mask==1 -> subsampled
//     else       -> bilinearPredict(subb, mask, y, x, factor)
__global__
void kernelReconstructTiles(const uchar3* __restrict__ prevRecon,
                            uchar3* __restrict__       outRecon,
                            const uchar3* __restrict__ subsampled,
                            const unsigned char* __restrict__ mask,
                            const unsigned char* __restrict__ tileActive,
                            int rows, int cols,
                            int tilesY, int tilesX,
                            int tileSize,
                            int factor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= cols || y >= rows) return;

    int idx = idx2d(y, x, cols);

    int ty = y / tileSize;
    int tx = x / tileSize;
    if (ty < 0 || ty >= tilesY || tx < 0 || tx >= tilesX) {
        outRecon[idx] = prevRecon[idx];
        return;
    }

    int tileIdx = ty * tilesX + tx;
    unsigned char active = tileActive[tileIdx];

    if (!active) {
        // static tile -> keep prevRecon
        outRecon[idx] = prevRecon[idx];
        return;
    }

    // Active tile
    if (mask[idx] == 1) {
        // known subsampled pixel
        outRecon[idx] = subsampled[idx];
        return;
    }

    // Missing pixel -> bilinearPredict(subb, mask, y, x, factor)
    int rowsSub = rows;
    int colsSub = cols;

    int x0 = (x / factor) * factor;
    int y0 = (y / factor) * factor;
    int x1 = x0 + factor;
    int y1 = y0 + factor;

    // Clamp to image bounds
    x0 = max(0, min(x0, colsSub - 1));
    y0 = max(0, min(y0, rowsSub - 1));
    x1 = max(0, min(x1, colsSub - 1));
    y1 = max(0, min(y1, rowsSub - 1));

    // Handle degenerate corner case
    int idx00 = idx2d(y0, x0, colsSub);
    if (x0 == x1 && y0 == y1 && mask[idx00] == 1) {
        outRecon[idx] = subsampled[idx00];
        return;
    }

    auto getColorIfSampled = [&](int yy, int xx, bool &has) -> float3 {
        int i = idx2d(yy, xx, colsSub);
        if (mask[i] == 1) {
            has = true;
            uchar3 c = subsampled[i];
            return make_float3((float)c.x, (float)c.y, (float)c.z);
        }
        has = false;
        return make_float3(0.f, 0.f, 0.f);
    };

    bool has00, has10, has01, has11;
    float3 C00 = getColorIfSampled(y0, x0, has00);
    float3 C10 = getColorIfSampled(y0, x1, has10);
    float3 C01 = getColorIfSampled(y1, x0, has01);
    float3 C11 = getColorIfSampled(y1, x1, has11);

    if (!has00 && !has10 && !has01 && !has11) {
        int nx = (x / factor) * factor;
        int ny = (y / factor) * factor;
        nx = max(0, min(nx, colsSub - 1));
        ny = max(0, min(ny, rowsSub - 1));
        int nidx = idx2d(ny, nx, colsSub);
        outRecon[idx] = subsampled[nidx];
        return;
    }

    double dx = (x1 == x0) ? 0.0 : double(x - x0) / double(x1 - x0);
    double dy = (y1 == y0) ? 0.0 : double(y - y0) / double(y1 - y0);

    double w00 = (1.0 - dx) * (1.0 - dy);
    double w10 = dx * (1.0 - dy);
    double w01 = (1.0 - dx) * dy;
    double w11 = dx * dy;

    double outB = 0.0, outG = 0.0, outR = 0.0;
    double wSum = 0.0;

    if (has00) { outB += w00 * C00.x; outG += w00 * C00.y; outR += w00 * C00.z; wSum += w00; }
    if (has10) { outB += w10 * C10.x; outG += w10 * C10.y; outR += w10 * C10.z; wSum += w10; }
    if (has01) { outB += w01 * C01.x; outG += w01 * C01.y; outR += w01 * C01.z; wSum += w01; }
    if (has11) { outB += w11 * C11.x; outG += w11 * C11.y; outR += w11 * C11.z; wSum += w11; }

    if (wSum > 0.0) {
        outB /= wSum;
        outG /= wSum;
        outR /= wSum;
    }

    auto clamp255 = [](double v) -> unsigned char {
        if (v < 0.0)   return 0;
        if (v > 255.0) return 255;
        return static_cast<unsigned char>(v + 0.5);
    };

    outRecon[idx] = make_uchar3(
        clamp255(outB),
        clamp255(outG),
        clamp255(outR)
    );
}

// -------------------- Persistent buffer manager --------------------

static void ensureDeviceBuffers(int rows, int cols,
                                int tilesY, int tilesX,
                                bool needTileMask)
{
    size_t numPixels = (size_t)rows * (size_t)cols;
    size_t numTiles  = (size_t)tilesY * (size_t)tilesX;

    // Pixel-sized buffers (img + subsampled + mask)
    if (numPixels > g_capacity_pixels) {
        auto t_free_start = Clock::now();
        if (g_d_img_cur)      cudaFree(g_d_img_cur);
        if (g_d_img_next)     cudaFree(g_d_img_next);
        if (g_d_subsampled)   cudaFree(g_d_subsampled);
        if (g_d_mask_dev)     cudaFree(g_d_mask_dev);
        auto t_free_end = Clock::now();
        if (g_capacity_pixels > 0) {
            g_free_ms += std::chrono::duration<double, std::milli>(
                             t_free_end - t_free_start).count();
        }

        size_t imgBytes   = sizeof(uchar3)        * numPixels;
        size_t maskBytes  = sizeof(unsigned char) * numPixels;

        auto t_malloc_start = Clock::now();
        checkCuda(cudaMalloc(&g_d_img_cur,      imgBytes),  "cudaMalloc g_d_img_cur");
        checkCuda(cudaMalloc(&g_d_img_next,     imgBytes),  "cudaMalloc g_d_img_next");
        checkCuda(cudaMalloc(&g_d_subsampled,   imgBytes),  "cudaMalloc g_d_subsampled");
        checkCuda(cudaMalloc(&g_d_mask_dev,     maskBytes), "cudaMalloc g_d_mask_dev");
        auto t_malloc_end = Clock::now();
        g_malloc_ms += std::chrono::duration<double, std::milli>(
                           t_malloc_end - t_malloc_start).count();

        g_capacity_pixels = numPixels;
    }

    // Tile-active buffer
    if (needTileMask) {
        if (numTiles > g_capacity_tiles) {
            auto t_free_start = Clock::now();
            if (g_d_tileActive_dev) {
                cudaFree(g_d_tileActive_dev);
            }
            auto t_free_end = Clock::now();
            if (g_capacity_tiles > 0) {
                g_free_ms += std::chrono::duration<double, std::milli>(
                                 t_free_end - t_free_start).count();
            }

            size_t tileBytes = sizeof(unsigned char) * numTiles;

            auto t_malloc_start = Clock::now();
            checkCuda(cudaMalloc(&g_d_tileActive_dev, tileBytes),
                      "cudaMalloc g_d_tileActive_dev");
            auto t_malloc_end = Clock::now();
            g_malloc_ms += std::chrono::duration<double, std::milli>(
                               t_malloc_end - t_malloc_start).count();

            g_capacity_tiles = numTiles;
        }
    }
}

// -------------------- Public APIs: refinement (unchanged) --------------------

void iterativeRefineCUDA(cv::Mat& img,
                         const cv::Mat& mask,
                         int iterations)
{
    if (!g_cuda_ok) {
        iterativeRefine(img, mask, iterations);
        return;
    }

    CV_Assert(img.size() == mask.size());
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(img.type() == CV_8UC3);
    CV_Assert(img.isContinuous());
    CV_Assert(mask.isContinuous());

    int rows = img.rows;
    int cols = img.cols;
    int numPixels = rows * cols;

    size_t imgBytes  = sizeof(uchar3)        * numPixels;
    size_t maskBytes = sizeof(unsigned char) * numPixels;

    ensureDeviceBuffers(rows, cols, 0, 0, false);

    cudaEvent_t eStartH2D, eEndH2D, eStartKernel, eEndKernel, eStartD2H, eEndD2H;
    cudaEventCreate(&eStartH2D);
    cudaEventCreate(&eEndH2D);
    cudaEventCreate(&eStartKernel);
    cudaEventCreate(&eEndKernel);
    cudaEventCreate(&eStartD2H);
    cudaEventCreate(&eEndD2H);

    cudaEventRecord(eStartH2D);
    checkCuda(cudaMemcpy(g_d_img_cur,
                         (const uchar3*)img.data,
                         imgBytes,
                         cudaMemcpyHostToDevice),
              "Memcpy img -> g_d_img_cur");
    checkCuda(cudaMemcpy(g_d_mask_dev,
                         mask.data,
                         maskBytes,
                         cudaMemcpyHostToDevice),
              "Memcpy mask -> g_d_mask_dev");
    cudaEventRecord(eEndH2D);
    cudaEventSynchronize(eEndH2D);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    cudaEventRecord(eStartKernel);
    uchar3* d_cur  = g_d_img_cur;
    uchar3* d_next = g_d_img_next;
    for (int it = 0; it < iterations; ++it) {
        kernelIterativeRefineFull<<<grid, block>>>(
            d_cur, d_next, g_d_mask_dev, rows, cols);
        checkCuda(cudaGetLastError(), "kernelIterativeRefineFull");
        std::swap(d_cur, d_next);
    }
    cudaEventRecord(eEndKernel);
    cudaEventSynchronize(eEndKernel);

    cudaEventRecord(eStartD2H);
    checkCuda(cudaMemcpy((uchar3*)img.data,
                         d_cur,
                         imgBytes,
                         cudaMemcpyDeviceToHost),
              "Memcpy d_cur -> img");
    cudaEventRecord(eEndD2H);
    cudaEventSynchronize(eEndD2H);

    float h2d_ms=0, k_ms=0, d2h_ms=0;
    cudaEventElapsedTime(&h2d_ms, eStartH2D, eEndH2D);
    cudaEventElapsedTime(&k_ms,   eStartKernel, eEndKernel);
    cudaEventElapsedTime(&d2h_ms, eStartD2H, eEndD2H);

    g_h2d_ms    += h2d_ms;
    g_kernel_ms += k_ms;
    g_d2h_ms    += d2h_ms;
    g_refine_calls++;

    cudaEventDestroy(eStartH2D);
    cudaEventDestroy(eEndH2D);
    cudaEventDestroy(eStartKernel);
    cudaEventDestroy(eEndKernel);
    cudaEventDestroy(eStartD2H);
    cudaEventDestroy(eEndD2H);
}

void iterativeRefineTilesCUDA(cv::Mat& img,
                              const cv::Mat& mask,
                              const cv::Mat& tileActiveMask,
                              int tileSize,
                              int iterations)
{
    if (!g_cuda_ok) {
        iterativeRefineTiles(img, mask, tileActiveMask, tileSize, iterations);
        return;
    }

    CV_Assert(img.size() == mask.size());
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(img.type() == CV_8UC3);
    CV_Assert(tileActiveMask.type() == CV_8UC1);
    CV_Assert(img.isContinuous());
    CV_Assert(mask.isContinuous());
    CV_Assert(tileActiveMask.isContinuous());

    int rows   = img.rows;
    int cols   = img.cols;
    int tilesY = tileActiveMask.rows;
    int tilesX = tileActiveMask.cols;

    int numPixels = rows * cols;
    int numTiles  = tilesY * tilesX;
    size_t imgBytes  = sizeof(uchar3)        * numPixels;
    size_t maskBytes = sizeof(unsigned char) * numPixels;
    size_t tileBytes = sizeof(unsigned char) * numTiles;

    ensureDeviceBuffers(rows, cols, tilesY, tilesX, true);

    cudaEvent_t eStartH2D, eEndH2D, eStartKernel, eEndKernel, eStartD2H, eEndD2H;
    cudaEventCreate(&eStartH2D);
    cudaEventCreate(&eEndH2D);
    cudaEventCreate(&eStartKernel);
    cudaEventCreate(&eEndKernel);
    cudaEventCreate(&eStartD2H);
    cudaEventCreate(&eEndD2H);

    cudaEventRecord(eStartH2D);
    checkCuda(cudaMemcpy(g_d_img_cur,
                         (const uchar3*)img.data,
                         imgBytes,
                         cudaMemcpyHostToDevice),
              "Memcpy img -> g_d_img_cur");
    checkCuda(cudaMemcpy(g_d_mask_dev,
                         mask.data,
                         maskBytes,
                         cudaMemcpyHostToDevice),
              "Memcpy mask -> g_d_mask_dev");
    checkCuda(cudaMemcpy(g_d_tileActive_dev,
                         tileActiveMask.data,
                         tileBytes,
                         cudaMemcpyHostToDevice),
              "Memcpy tileActive -> g_d_tileActive_dev");
    cudaEventRecord(eEndH2D);
    cudaEventSynchronize(eEndH2D);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    cudaEventRecord(eStartKernel);
    uchar3* d_cur  = g_d_img_cur;
    uchar3* d_next = g_d_img_next;
    for (int it = 0; it < iterations; ++it) {
        kernelIterativeRefineTiles<<<grid, block>>>(
            d_cur, d_next,
            g_d_mask_dev,
            g_d_tileActive_dev,
            rows, cols,
            tilesY, tilesX,
            tileSize);
        checkCuda(cudaGetLastError(), "kernelIterativeRefineTiles");
        std::swap(d_cur, d_next);
    }
    cudaEventRecord(eEndKernel);
    cudaEventSynchronize(eEndKernel);

    cudaEventRecord(eStartD2H);
    checkCuda(cudaMemcpy((uchar3*)img.data,
                         d_cur,
                         imgBytes,
                         cudaMemcpyDeviceToHost),
              "Memcpy d_cur -> img");
    cudaEventRecord(eEndD2H);
    cudaEventSynchronize(eEndD2H);

    float h2d_ms=0, k_ms=0, d2h_ms=0;
    cudaEventElapsedTime(&h2d_ms, eStartH2D, eEndH2D);
    cudaEventElapsedTime(&k_ms,   eStartKernel, eEndKernel);
    cudaEventElapsedTime(&d2h_ms, eStartD2H, eEndD2H);

    g_h2d_ms        += h2d_ms;
    g_kernel_ms     += k_ms;
    g_d2h_ms        += d2h_ms;
    g_refine_tiles_calls++;

    cudaEventDestroy(eStartH2D);
    cudaEventDestroy(eEndH2D);
    cudaEventDestroy(eStartKernel);
    cudaEventDestroy(eEndKernel);
    cudaEventDestroy(eStartD2H);
    cudaEventDestroy(eEndD2H);
}

// -------------------- Public API: reconstruct tiles (new) --------------------

void reconstructTilesCUDA(cv::Mat& finalRecon,
                          const cv::Mat& prevRecon,
                          const cv::Mat& subsampled,
                          const cv::Mat& mask,
                          const cv::Mat& tileActiveMask,
                          int tileSize,
                          int factor)
{
    if (!g_cuda_ok) {
        // fallback: just do what you already do on CPU
        finalRecon = prevRecon.clone();
        int tilesY = tileActiveMask.rows;
        int tilesX = tileActiveMask.cols;
        int rows   = finalRecon.rows;
        int cols   = finalRecon.cols;

        #pragma omp parallel for schedule(static)
        for (int ty = 0; ty < tilesY; ++ty) {
            for (int tx = 0; tx < tilesX; ++tx) {
                uint8_t active = tileActiveMask.at<uint8_t>(ty, tx);
                if (!active) continue;

                int y0 = ty * tileSize;
                int x0 = tx * tileSize;
                int y1 = std::min(y0 + tileSize, rows);
                int x1 = std::min(x0 + tileSize, cols);

                for (int y = y0; y < y1; ++y) {
                    for (int x = x0; x < x1; ++x) {
                        if (mask.at<uint8_t>(y, x) == 1) {
                            finalRecon.at<cv::Vec3b>(y, x) =
                                subsampled.at<cv::Vec3b>(y, x);
                        } else {
                            finalRecon.at<cv::Vec3b>(y, x) =
                                bilinearPredict(subsampled, mask, y, x, factor);
                        }
                    }
                }
            }
        }
        return;
    }

    CV_Assert(prevRecon.size() == subsampled.size());
    CV_Assert(prevRecon.size() == mask.size());
    CV_Assert(prevRecon.type() == CV_8UC3);
    CV_Assert(subsampled.type() == CV_8UC3);
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(tileActiveMask.type() == CV_8UC1);
    CV_Assert(prevRecon.isContinuous());
    CV_Assert(subsampled.isContinuous());
    CV_Assert(mask.isContinuous());
    CV_Assert(tileActiveMask.isContinuous());

    int rows   = prevRecon.rows;
    int cols   = prevRecon.cols;
    int tilesY = tileActiveMask.rows;
    int tilesX = tileActiveMask.cols;

    int numPixels = rows * cols;
    int numTiles  = tilesY * tilesX;

    size_t imgBytes  = sizeof(uchar3)        * numPixels;
    size_t maskBytes = sizeof(unsigned char) * numPixels;
    size_t tileBytes = sizeof(unsigned char) * numTiles;

    ensureDeviceBuffers(rows, cols, tilesY, tilesX, true);

    cudaEvent_t eStartH2D, eEndH2D, eStartKernel, eEndKernel, eStartD2H, eEndD2H;
    cudaEventCreate(&eStartH2D);
    cudaEventCreate(&eEndH2D);
    cudaEventCreate(&eStartKernel);
    cudaEventCreate(&eEndKernel);
    cudaEventCreate(&eStartD2H);
    cudaEventCreate(&eEndD2H);

    // H2D
    cudaEventRecord(eStartH2D);
    checkCuda(cudaMemcpy(g_d_img_cur,
                         (const uchar3*)prevRecon.data,
                         imgBytes,
                         cudaMemcpyHostToDevice),
              "Memcpy prevRecon -> g_d_img_cur");
    checkCuda(cudaMemcpy(g_d_subsampled,
                         (const uchar3*)subsampled.data,
                         imgBytes,
                         cudaMemcpyHostToDevice),
              "Memcpy subsampled -> g_d_subsampled");
    checkCuda(cudaMemcpy(g_d_mask_dev,
                         mask.data,
                         maskBytes,
                         cudaMemcpyHostToDevice),
              "Memcpy mask -> g_d_mask_dev");
    checkCuda(cudaMemcpy(g_d_tileActive_dev,
                         tileActiveMask.data,
                         tileBytes,
                         cudaMemcpyHostToDevice),
              "Memcpy tileActive -> g_d_tileActive_dev");
    cudaEventRecord(eEndH2D);
    cudaEventSynchronize(eEndH2D);

    // Kernel
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    cudaEventRecord(eStartKernel);
    kernelReconstructTiles<<<grid, block>>>(
        g_d_img_cur,
        g_d_img_next,
        g_d_subsampled,
        g_d_mask_dev,
        g_d_tileActive_dev,
        rows, cols,
        tilesY, tilesX,
        tileSize,
        factor);
    checkCuda(cudaGetLastError(), "kernelReconstructTiles");
    cudaEventRecord(eEndKernel);
    cudaEventSynchronize(eEndKernel);

    // D2H
    cudaEventRecord(eStartD2H);
    if (finalRecon.empty() || finalRecon.rows != rows || finalRecon.cols != cols) {
        finalRecon.create(rows, cols, CV_8UC3);
    }
    checkCuda(cudaMemcpy((uchar3*)finalRecon.data,
                         g_d_img_next,
                         imgBytes,
                         cudaMemcpyDeviceToHost),
              "Memcpy g_d_img_next -> finalRecon");
    cudaEventRecord(eEndD2H);
    cudaEventSynchronize(eEndD2H);

    float h2d_ms=0, k_ms=0, d2h_ms=0;
    cudaEventElapsedTime(&h2d_ms, eStartH2D, eEndH2D);
    cudaEventElapsedTime(&k_ms,   eStartKernel, eEndKernel);
    cudaEventElapsedTime(&d2h_ms, eStartD2H, eEndD2H);

    g_h2d_ms        += h2d_ms;
    g_kernel_ms     += k_ms;
    g_d2h_ms        += d2h_ms;
    g_recon_tiles_calls++;

    cudaEventDestroy(eStartH2D);
    cudaEventDestroy(eEndH2D);
    cudaEventDestroy(eStartKernel);
    cudaEventDestroy(eEndKernel);
    cudaEventDestroy(eStartD2H);
    cudaEventDestroy(eEndD2H);
}

// -------------------- Stats printing --------------------

void cudaPrintRefineStats()
{
    std::cout << "[CUDA refine stats]\n";
    std::cout << "  device          : " << g_cuda_device_name << "\n";
    std::cout << "  refine calls    : " << g_refine_calls
              << " (full-frame)\n";
    std::cout << "  refineTiles calls: " << g_refine_tiles_calls
              << " (tile-aware)\n";
    std::cout << "  reconTiles calls: " << g_recon_tiles_calls
              << " (tile-aware)\n";
    std::cout << "  total cudaMalloc: " << g_malloc_ms << " ms\n";
    std::cout << "  total cudaFree  : " << g_free_ms << " ms\n";
    std::cout << "  total H2D       : " << g_h2d_ms << " ms\n";
    std::cout << "  total kernel    : " << g_kernel_ms << " ms\n";
    std::cout << "  total D2H       : " << g_d2h_ms << " ms\n";

    int totalCalls = g_refine_calls + g_refine_tiles_calls + g_recon_tiles_calls;
    if (totalCalls > 0) {
        double avgMallocFree =
            (g_malloc_ms + g_free_ms) / double(totalCalls);
        double avgH2D =
            g_h2d_ms / double(totalCalls);
        double avgKernel =
            g_kernel_ms / double(totalCalls);
        double avgD2H =
            g_d2h_ms / double(totalCalls);

        std::cout << "  avg per call (malloc+free): " << avgMallocFree << " ms\n";
        std::cout << "  avg per call H2D          : " << avgH2D        << " ms\n";
        std::cout << "  avg per call kernel       : " << avgKernel     << " ms\n";
        std::cout << "  avg per call D2H          : " << avgD2H        << " ms\n";
    }
    std::cout << "--------------------------------------------------\n";
}
