// recon_cuda.cu
#include "recon_cuda.h"
#include "recon.h"        // for CPU fallback: iterativeRefine, iterativeRefineTiles

#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <chrono>

using Clock = std::chrono::high_resolution_clock;

// -------------------- Global CUDA state --------------------

static bool   g_cuda_initialized = false;
static bool   g_cuda_ok          = false;
static int    g_cuda_device_id   = -1;
static char   g_cuda_device_name[256] = {0};

// -------------------- Profiling accumulators --------------------
// All in milliseconds
static double g_malloc_ms      = 0.0;
static double g_free_ms        = 0.0;
static double g_h2d_ms         = 0.0;
static double g_kernel_ms      = 0.0;
static double g_d2h_ms         = 0.0;

static int    g_refine_calls       = 0;
static int    g_refine_tiles_calls = 0;

// -------------------- Helpers --------------------

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] " << msg << ": "
                  << cudaGetErrorString(err) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

// Initialize CUDA once, cache result.
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

    g_cuda_device_id = 0; // pick device 0 for now
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, g_cuda_device_id) == cudaSuccess) {
        std::snprintf(g_cuda_device_name, sizeof(g_cuda_device_name),
                      "%s (cc %d.%d, %d SMs)",
                      prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    } else {
        std::snprintf(g_cuda_device_name, sizeof(g_cuda_device_name),
                      "unknown device");
    }

    cudaSetDevice(g_cuda_device_id);
    std::cerr << "[CUDA] Using device " << g_cuda_device_id
              << ": " << g_cuda_device_name << "\n";

    g_cuda_ok = true;
    return true;
}

// -------------------- Device helpers ----------------------------

__device__ inline int idx2d(int y, int x, int cols) {
    return y * cols + x;
}

// Full-frame refine kernel (for iterativeRefine)
__global__
void kernelIterativeRefine(const uchar3* cur,
                           uchar3* next,
                           const unsigned char* mask,
                           int rows, int cols)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows) return;

    int idx = idx2d(y, x, cols);

    // Known pixels stay fixed
    if (mask[idx] == 1) {
        next[idx] = cur[idx];
        return;
    }

    // 3x3 average, same as average3x3MissingAware
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
        return;
    }

    uchar3 out;
    out.x = static_cast<unsigned char>(sumB / count);
    out.y = static_cast<unsigned char>(sumG / count);
    out.z = static_cast<unsigned char>(sumR / count);
    next[idx] = out;
}

// Tile-aware refine kernel (for iterativeRefineTiles)
__global__
void kernelIterativeRefineTiles(const uchar3* cur,
                                uchar3* next,
                                const unsigned char* mask,
                                const unsigned char* tileActive,
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
    if (ty >= tilesY) ty = tilesY - 1;
    if (tx >= tilesX) tx = tilesX - 1;

    int tidx = idx2d(ty, tx, tilesX);
    unsigned char active = tileActive[tidx];

    // if tile is inactive or pixel is known -> copy
    if (!active || mask[idx] == 1) {
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
        return;
    }

    uchar3 out;
    out.x = static_cast<unsigned char>(sumB / count);
    out.y = static_cast<unsigned char>(sumG / count);
    out.z = static_cast<unsigned char>(sumR / count);
    next[idx] = out;
}

// -------------------- Host wrappers with timing -----------------

void iterativeRefineCUDA(cv::Mat& img,
                         const cv::Mat& mask,
                         int iterations)
{
    // If CUDA not available, fall back to CPU implementation
    if (!cudaRefinementInit()) {
        iterativeRefine(img, mask, iterations);
        return;
    }

    CV_Assert(img.type()  == CV_8UC3);
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(img.size()  == mask.size());
    CV_Assert(img.isContinuous());
    CV_Assert(mask.isContinuous());

    int rows = img.rows;
    int cols = img.cols;
    int numPixels = rows * cols;

    uchar3*        d_cur  = nullptr;
    uchar3*        d_next = nullptr;
    unsigned char* d_mask = nullptr;

    size_t imgBytes  = sizeof(uchar3)        * numPixels;
    size_t maskBytes = sizeof(unsigned char) * numPixels;

    // Time cudaMalloc as host-side cost
    auto t_malloc_start = Clock::now();
    checkCuda(cudaMalloc(&d_cur,  imgBytes),  "cudaMalloc d_cur");
    checkCuda(cudaMalloc(&d_next, imgBytes),  "cudaMalloc d_next");
    checkCuda(cudaMalloc(&d_mask, maskBytes), "cudaMalloc d_mask");
    auto t_malloc_end = Clock::now();
    g_malloc_ms += std::chrono::duration<double, std::milli>(t_malloc_end - t_malloc_start).count();

    // CUDA events for device-side timing (H2D, kernel, D2H)
    cudaEvent_t eStartH2D, eEndH2D, eStartKernel, eEndKernel, eStartD2H, eEndD2H;
    cudaEventCreate(&eStartH2D);
    cudaEventCreate(&eEndH2D);
    cudaEventCreate(&eStartKernel);
    cudaEventCreate(&eEndKernel);
    cudaEventCreate(&eStartD2H);
    cudaEventCreate(&eEndD2H);

    // H2D copies
    cudaEventRecord(eStartH2D);
    checkCuda(cudaMemcpy(d_cur,
                         reinterpret_cast<uchar3*>(img.data),
                         imgBytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy img -> d_cur");
    checkCuda(cudaMemcpy(d_mask,
                         mask.data,
                         maskBytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy mask -> d_mask");
    cudaEventRecord(eEndH2D);
    cudaEventSynchronize(eEndH2D);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    // Kernels loop
    cudaEventRecord(eStartKernel);
    for (int it = 0; it < iterations; ++it) {
        kernelIterativeRefine<<<grid, block>>>(d_cur, d_next, d_mask, rows, cols);
        checkCuda(cudaGetLastError(), "kernelIterativeRefine launch");
        checkCuda(cudaDeviceSynchronize(), "kernelIterativeRefine sync");

        uchar3* tmp = d_cur;
        d_cur  = d_next;
        d_next = tmp;
    }
    cudaEventRecord(eEndKernel);
    cudaEventSynchronize(eEndKernel);

    // D2H copy
    cudaEventRecord(eStartD2H);
    checkCuda(cudaMemcpy(reinterpret_cast<uchar3*>(img.data),
                         d_cur,
                         imgBytes,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy d_cur -> img");
    cudaEventRecord(eEndD2H);
    cudaEventSynchronize(eEndD2H);

    // Collect timings
    float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms,    eStartH2D,    eEndH2D);
    cudaEventElapsedTime(&kernel_ms, eStartKernel, eEndKernel);
    cudaEventElapsedTime(&d2h_ms,    eStartD2H,    eEndD2H);

    g_h2d_ms    += h2d_ms;
    g_kernel_ms += kernel_ms;
    g_d2h_ms    += d2h_ms;
    g_refine_calls++;

    cudaEventDestroy(eStartH2D);
    cudaEventDestroy(eEndH2D);
    cudaEventDestroy(eStartKernel);
    cudaEventDestroy(eEndKernel);
    cudaEventDestroy(eStartD2H);
    cudaEventDestroy(eEndD2H);

    // Time cudaFree as host-side cost
    auto t_free_start = Clock::now();
    cudaFree(d_cur);
    cudaFree(d_next);
    cudaFree(d_mask);
    auto t_free_end = Clock::now();
    g_free_ms += std::chrono::duration<double, std::milli>(t_free_end - t_free_start).count();
}

void iterativeRefineTilesCUDA(cv::Mat& img,
                              const cv::Mat& mask,
                              const cv::Mat& tileActiveMask,
                              int tileSize,
                              int iterations)
{
    if (!cudaRefinementInit()) {
        iterativeRefineTiles(img, mask, tileActiveMask, tileSize, iterations);
        return;
    }

    CV_Assert(img.type()           == CV_8UC3);
    CV_Assert(mask.type()          == CV_8UC1);
    CV_Assert(tileActiveMask.type()== CV_8UC1);
    CV_Assert(img.size()           == mask.size());
    CV_Assert(img.isContinuous());
    CV_Assert(mask.isContinuous());
    CV_Assert(tileActiveMask.isContinuous());

    int rows   = img.rows;
    int cols   = img.cols;
    int tilesY = tileActiveMask.rows;
    int tilesX = tileActiveMask.cols;

    int numPixels = rows * cols;
    int numTiles  = tilesY * tilesX;

    uchar3*        d_cur        = nullptr;
    uchar3*        d_next       = nullptr;
    unsigned char* d_mask       = nullptr;
    unsigned char* d_tileActive = nullptr;

    size_t imgBytes  = sizeof(uchar3)        * numPixels;
    size_t maskBytes = sizeof(unsigned char) * numPixels;
    size_t tileBytes = sizeof(unsigned char) * numTiles;

    auto t_malloc_start = Clock::now();
    checkCuda(cudaMalloc(&d_cur,        imgBytes),   "cudaMalloc d_cur");
    checkCuda(cudaMalloc(&d_next,       imgBytes),   "cudaMalloc d_next");
    checkCuda(cudaMalloc(&d_mask,       maskBytes),  "cudaMalloc d_mask");
    checkCuda(cudaMalloc(&d_tileActive, tileBytes),  "cudaMalloc d_tileActive");
    auto t_malloc_end = Clock::now();
    g_malloc_ms += std::chrono::duration<double, std::milli>(t_malloc_end - t_malloc_start).count();

    cudaEvent_t eStartH2D, eEndH2D, eStartKernel, eEndKernel, eStartD2H, eEndD2H;
    cudaEventCreate(&eStartH2D);
    cudaEventCreate(&eEndH2D);
    cudaEventCreate(&eStartKernel);
    cudaEventCreate(&eEndKernel);
    cudaEventCreate(&eStartD2H);
    cudaEventCreate(&eEndD2H);

    // H2D
    cudaEventRecord(eStartH2D);
    checkCuda(cudaMemcpy(d_cur,
                         reinterpret_cast<uchar3*>(img.data),
                         imgBytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy img -> d_cur");
    checkCuda(cudaMemcpy(d_mask,
                         mask.data,
                         maskBytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy mask -> d_mask");
    checkCuda(cudaMemcpy(d_tileActive,
                         tileActiveMask.data,
                         tileBytes,
                         cudaMemcpyHostToDevice),
              "cudaMemcpy tileActive -> d_tileActive");
    cudaEventRecord(eEndH2D);
    cudaEventSynchronize(eEndH2D);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    // Kernels
    cudaEventRecord(eStartKernel);
    for (int it = 0; it < iterations; ++it) {
        kernelIterativeRefineTiles<<<grid, block>>>(
            d_cur, d_next, d_mask, d_tileActive,
            rows, cols, tilesY, tilesX, tileSize
        );
        checkCuda(cudaGetLastError(), "kernelIterativeRefineTiles launch");
        checkCuda(cudaDeviceSynchronize(), "kernelIterativeRefineTiles sync");

        uchar3* tmp = d_cur;
        d_cur  = d_next;
        d_next = tmp;
    }
    cudaEventRecord(eEndKernel);
    cudaEventSynchronize(eEndKernel);

    // D2H
    cudaEventRecord(eStartD2H);
    checkCuda(cudaMemcpy(reinterpret_cast<uchar3*>(img.data),
                         d_cur,
                         imgBytes,
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy d_cur -> img");
    cudaEventRecord(eEndD2H);
    cudaEventSynchronize(eEndD2H);

    float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms,    eStartH2D,    eEndH2D);
    cudaEventElapsedTime(&kernel_ms, eStartKernel, eEndKernel);
    cudaEventElapsedTime(&d2h_ms,    eStartD2H,    eEndD2H);

    g_h2d_ms    += h2d_ms;
    g_kernel_ms += kernel_ms;
    g_d2h_ms    += d2h_ms;
    g_refine_tiles_calls++;

    cudaEventDestroy(eStartH2D);
    cudaEventDestroy(eEndH2D);
    cudaEventDestroy(eStartKernel);
    cudaEventDestroy(eEndKernel);
    cudaEventDestroy(eStartD2H);
    cudaEventDestroy(eEndD2H);

    auto t_free_start = Clock::now();
    cudaFree(d_cur);
    cudaFree(d_next);
    cudaFree(d_mask);
    cudaFree(d_tileActive);
    auto t_free_end = Clock::now();
    g_free_ms += std::chrono::duration<double, std::milli>(t_free_end - t_free_start).count();
}

// -------------------- Stats print -----------------------------

void cudaPrintRefineStats() {
    std::cerr << "\n[CUDA refine stats]\n";
    std::cerr << "  device          : "
              << (g_cuda_device_name[0] ? g_cuda_device_name : "none/unknown")
              << "\n";
    std::cerr << "  refine calls    : " << g_refine_calls
              << " (full-frame)\n";
    std::cerr << "  refineTiles calls: " << g_refine_tiles_calls
              << " (tile-aware)\n";

    std::cerr << "  total cudaMalloc: " << g_malloc_ms << " ms\n";
    std::cerr << "  total cudaFree  : " << g_free_ms   << " ms\n";
    std::cerr << "  total H2D       : " << g_h2d_ms    << " ms\n";
    std::cerr << "  total kernel    : " << g_kernel_ms << " ms\n";
    std::cerr << "  total D2H       : " << g_d2h_ms    << " ms\n";

    int total_calls = g_refine_calls + g_refine_tiles_calls;
    if (total_calls > 0) {
        std::cerr << "  avg per call (malloc+free): "
                  << (g_malloc_ms + g_free_ms) / total_calls << " ms\n";
        std::cerr << "  avg per call H2D          : "
                  << g_h2d_ms / total_calls << " ms\n";
        std::cerr << "  avg per call kernel       : "
                  << g_kernel_ms / total_calls << " ms\n";
        std::cerr << "  avg per call D2H          : "
                  << g_d2h_ms / total_calls << " ms\n";
    }
    std::cerr << "--------------------------------------------------\n";
}