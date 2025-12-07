// recon_cuda.cu
#include "recon_cuda.h"
#include "recon.h"        // for CPU fallback: iterativeRefine, iterativeRefineTiles

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

// -------------------- Profiling accumulators --------------------
// All in milliseconds

static double g_malloc_ms      = 0.0;
static double g_free_ms        = 0.0;
static double g_h2d_ms         = 0.0;
static double g_kernel_ms      = 0.0;
static double g_d2h_ms         = 0.0;

static int    g_refine_calls       = 0;
static int    g_refine_tiles_calls = 0;

// -------------------- Persistent device buffers --------------------
// Reused across frames to avoid per-call cudaMalloc/cudaFree overhead.

static uchar3*        g_d_img_cur        = nullptr;
static uchar3*        g_d_img_next       = nullptr;
static unsigned char* g_d_mask_dev       = nullptr;
static unsigned char* g_d_tileActive_dev = nullptr;

static size_t g_capacity_pixels = 0;  // number of pixels these buffers can hold
static size_t g_capacity_tiles  = 0;  // number of tiles tileActive buffer can hold

// -------------------- Error helper ----------------------------

static inline void checkCuda(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::cerr << "[CUDA] Error at " << what << ": "
                  << cudaGetErrorString(err) << "\n";
        throw std::runtime_error("CUDA error");
    }
}

// -------------------- Init / teardown ----------------------------

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

    // Reset profiling accumulators
    g_malloc_ms = g_free_ms = g_h2d_ms = g_kernel_ms = g_d2h_ms = 0.0;
    g_refine_calls = g_refine_tiles_calls = 0;

    g_capacity_pixels = 0;
    g_capacity_tiles  = 0;

    g_cuda_ok = true;
    return true;
}

// -------------------- Device helpers ----------------------------

__device__ inline int idx2d(int y, int x, int cols) {
    return y * cols + x;
}

// -------------------- Kernels --------------------

// Full-frame refinement kernel (for iterativeRefine)
// Mirrors average3x3MissingAware + anchor behavior:
// - If mask[y,x] == 1: keep original pixel.
// - Else: average 3x3 neighborhood from cur and write to next.
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
        // Known pixels stay fixed
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

// Tile-aware refinement kernel (for iterativeRefineTiles)
// Only operates on tiles marked active in tileActiveMask.
// Behavior within an active tile is the same as the full-frame kernel;
// inactive tiles simply copy cur -> next.
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
        // Inactive tile: just copy
        next[idx] = cur[idx];
        return;
    }

    if (mask[idx] == 1) {
        // Known pixels stay fixed
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

// -------------------- Persistent buffer manager --------------------

// Ensure device buffers are allocated for at least (rows*cols) pixels
// and (tilesY*tilesX) tiles if needTileMask == true.
// We also attribute malloc/free time to g_malloc_ms / g_free_ms.
static void ensureDeviceBuffers(int rows, int cols,
                                int tilesY, int tilesX,
                                bool needTileMask)
{
    const size_t numPixels = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t numTiles  = static_cast<size_t>(tilesY) * static_cast<size_t>(tilesX);

    // --- Pixel-sized buffers (img + mask) ---
    if (numPixels > g_capacity_pixels) {
        auto t_free_start = Clock::now();
        if (g_d_img_cur)  cudaFree(g_d_img_cur);
        if (g_d_img_next) cudaFree(g_d_img_next);
        if (g_d_mask_dev) cudaFree(g_d_mask_dev);
        auto t_free_end = Clock::now();
        if (g_capacity_pixels > 0) {
            g_free_ms += std::chrono::duration<double, std::milli>(
                             t_free_end - t_free_start).count();
        }

        const size_t imgBytes  = sizeof(uchar3)        * numPixels;
        const size_t maskBytes = sizeof(unsigned char) * numPixels;

        auto t_malloc_start = Clock::now();
        checkCuda(cudaMalloc(&g_d_img_cur,  imgBytes),  "cudaMalloc g_d_img_cur");
        checkCuda(cudaMalloc(&g_d_img_next, imgBytes),  "cudaMalloc g_d_img_next");
        checkCuda(cudaMalloc(&g_d_mask_dev, maskBytes), "cudaMalloc g_d_mask_dev");
        auto t_malloc_end = Clock::now();
        g_malloc_ms += std::chrono::duration<double, std::milli>(
                           t_malloc_end - t_malloc_start).count();

        g_capacity_pixels = numPixels;
    }

    // --- Tile-active mask buffer ---
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

            const size_t tileBytes = sizeof(unsigned char) * numTiles;

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

// -------------------- Public CUDA refinement APIs --------------------

// Full-frame refinement (frame 0)
// Mirrors iterativeRefine on CPU, but using CUDA kernels.
void iterativeRefineCUDA(cv::Mat& img,
                         const cv::Mat& mask,
                         int iterations)
{
    if (!g_cuda_ok) {
        // Fallback to CPU implementation
        iterativeRefine(img, mask, iterations);
        return;
    }

    CV_Assert(img.size() == mask.size());
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(img.type() == CV_8UC3);
    CV_Assert(img.isContinuous());
    CV_Assert(mask.isContinuous());

    const int rows = img.rows;
    const int cols = img.cols;
    const int numPixels = rows * cols;

    const size_t imgBytes  = sizeof(uchar3)        * numPixels;
    const size_t maskBytes = sizeof(unsigned char) * numPixels;

    // Ensure device buffers exist / are large enough
    ensureDeviceBuffers(rows, cols, /*tilesY=*/0, /*tilesX=*/0, /*needTileMask=*/false);

    // CUDA events for device-side timing
    cudaEvent_t eStartH2D, eEndH2D, eStartKernel, eEndKernel, eStartD2H, eEndD2H;
    cudaEventCreate(&eStartH2D);
    cudaEventCreate(&eEndH2D);
    cudaEventCreate(&eStartKernel);
    cudaEventCreate(&eEndKernel);
    cudaEventCreate(&eStartD2H);
    cudaEventCreate(&eEndD2H);

    // ---------------- H2D copies ----------------
    cudaEventRecord(eStartH2D);
    checkCuda(cudaMemcpy(
                  g_d_img_cur,
                  reinterpret_cast<const uchar3*>(img.data),
                  imgBytes,
                  cudaMemcpyHostToDevice),
              "cudaMemcpy img -> g_d_img_cur");
    checkCuda(cudaMemcpy(
                  g_d_mask_dev,
                  mask.data,
                  maskBytes,
                  cudaMemcpyHostToDevice),
              "cudaMemcpy mask -> g_d_mask_dev");
    cudaEventRecord(eEndH2D);
    cudaEventSynchronize(eEndH2D);

    // ---------------- Kernel loop ----------------
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    cudaEventRecord(eStartKernel);

    uchar3* d_cur  = g_d_img_cur;
    uchar3* d_next = g_d_img_next;

    for (int it = 0; it < iterations; ++it) {
        kernelIterativeRefineFull<<<grid, block>>>(
            d_cur,
            d_next,
            g_d_mask_dev,
            rows,
            cols
        );
        checkCuda(cudaGetLastError(), "kernelIterativeRefineFull launch");
        std::swap(d_cur, d_next);
    }

    cudaEventRecord(eEndKernel);
    cudaEventSynchronize(eEndKernel);

    // ---------------- D2H copy ----------------
    cudaEventRecord(eStartD2H);
    checkCuda(cudaMemcpy(
                  reinterpret_cast<uchar3*>(img.data),
                  d_cur,
                  imgBytes,
                  cudaMemcpyDeviceToHost),
              "cudaMemcpy d_cur -> img");
    cudaEventRecord(eEndD2H);
    cudaEventSynchronize(eEndD2H);

    // ---------------- Accumulate timing + cleanup events ----------------
    float h2d_ms    = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms    = 0.0f;

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
}

// Tile-aware refinement (frames > 0)
// Mirrors iterativeRefineTiles on CPU, using tileActiveMask.
void iterativeRefineTilesCUDA(cv::Mat& img,
                              const cv::Mat& mask,
                              const cv::Mat& tileActiveMask,
                              int tileSize,
                              int iterations)
{
    if (!g_cuda_ok) {
        // Fallback to CPU implementation
        iterativeRefineTiles(img, mask, tileActiveMask, tileSize, iterations);
        return;
    }

    CV_Assert(img.size()            == mask.size());
    CV_Assert(mask.type()           == CV_8UC1);
    CV_Assert(img.type()            == CV_8UC3);
    CV_Assert(tileActiveMask.type() == CV_8UC1);
    CV_Assert(img.isContinuous());
    CV_Assert(mask.isContinuous());
    CV_Assert(tileActiveMask.isContinuous());

    const int rows   = img.rows;
    const int cols   = img.cols;
    const int tilesY = tileActiveMask.rows;
    const int tilesX = tileActiveMask.cols;

    const int    numPixels = rows * cols;
    const int    numTiles  = tilesY * tilesX;
    const size_t imgBytes  = sizeof(uchar3)        * numPixels;
    const size_t maskBytes = sizeof(unsigned char) * numPixels;
    const size_t tileBytes = sizeof(unsigned char) * numTiles;

    // Ensure device buffers exist / are large enough, including tiles
    ensureDeviceBuffers(rows, cols, tilesY, tilesX, /*needTileMask=*/true);

    // CUDA events
    cudaEvent_t eStartH2D, eEndH2D, eStartKernel, eEndKernel, eStartD2H, eEndD2H;
    cudaEventCreate(&eStartH2D);
    cudaEventCreate(&eEndH2D);
    cudaEventCreate(&eStartKernel);
    cudaEventCreate(&eEndKernel);
    cudaEventCreate(&eStartD2H);
    cudaEventCreate(&eEndD2H);

    // ---------------- H2D copies ----------------
    cudaEventRecord(eStartH2D);

    checkCuda(cudaMemcpy(
                  g_d_img_cur,
                  reinterpret_cast<const uchar3*>(img.data),
                  imgBytes,
                  cudaMemcpyHostToDevice),
              "cudaMemcpy img -> g_d_img_cur");
    checkCuda(cudaMemcpy(
                  g_d_mask_dev,
                  mask.data,
                  maskBytes,
                  cudaMemcpyHostToDevice),
              "cudaMemcpy mask -> g_d_mask_dev");
    checkCuda(cudaMemcpy(
                  g_d_tileActive_dev,
                  tileActiveMask.data,
                  tileBytes,
                  cudaMemcpyHostToDevice),
              "cudaMemcpy tileActiveMask -> g_d_tileActive_dev");

    cudaEventRecord(eEndH2D);
    cudaEventSynchronize(eEndH2D);

    // ---------------- Kernel loop ----------------
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);

    cudaEventRecord(eStartKernel);

    uchar3* d_cur  = g_d_img_cur;
    uchar3* d_next = g_d_img_next;

    for (int it = 0; it < iterations; ++it) {
        kernelIterativeRefineTiles<<<grid, block>>>(
            d_cur,
            d_next,
            g_d_mask_dev,
            g_d_tileActive_dev,
            rows,
            cols,
            tilesY,
            tilesX,
            tileSize
        );
        checkCuda(cudaGetLastError(), "kernelIterativeRefineTiles launch");
        std::swap(d_cur, d_next);
    }

    cudaEventRecord(eEndKernel);
    cudaEventSynchronize(eEndKernel);

    // ---------------- D2H copy ----------------
    cudaEventRecord(eStartD2H);
    checkCuda(cudaMemcpy(
                  reinterpret_cast<uchar3*>(img.data),
                  d_cur,
                  imgBytes,
                  cudaMemcpyDeviceToHost),
              "cudaMemcpy d_cur -> img");
    cudaEventRecord(eEndD2H);
    cudaEventSynchronize(eEndD2H);

    // ---------------- Accumulate timing ----------------
    float h2d_ms    = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms    = 0.0f;

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
    std::cout << "  total cudaMalloc: " << g_malloc_ms << " ms\n";
    std::cout << "  total cudaFree  : " << g_free_ms << " ms\n";
    std::cout << "  total H2D       : " << g_h2d_ms << " ms\n";
    std::cout << "  total kernel    : " << g_kernel_ms << " ms\n";
    std::cout << "  total D2H       : " << g_d2h_ms << " ms\n";

    int totalCalls = g_refine_calls + g_refine_tiles_calls;
    if (totalCalls > 0) {
        double avgMallocFree =
            (g_malloc_ms + g_free_ms) / static_cast<double>(totalCalls);
        double avgH2D =
            g_h2d_ms / static_cast<double>(totalCalls);
        double avgKernel =
            g_kernel_ms / static_cast<double>(totalCalls);
        double avgD2H =
            g_d2h_ms / static_cast<double>(totalCalls);

        std::cout << "  avg per call (malloc+free): " << avgMallocFree << " ms\n";
        std::cout << "  avg per call H2D          : " << avgH2D        << " ms\n";
        std::cout << "  avg per call kernel       : " << avgKernel     << " ms\n";
        std::cout << "  avg per call D2H          : " << avgD2H        << " ms\n";
    }
    std::cout << "--------------------------------------------------\n";
}
