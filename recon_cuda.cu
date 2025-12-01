// recon_cuda.cu
#include "recon_cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>

// Convert (y,x) to linear index
__device__ inline int idx2d(int y, int x, int width) {
    return y * width + x;
}

// GPU bilinearPredict equivalent
__device__ uchar3 bilinearPredictDevice(const uchar3* subsampled,
                                        const unsigned char* mask,
                                        int width, int height,
                                        int y, int x,
                                        int factor)
{
    int x0 = (x / factor) * factor;
    int y0 = (y / factor) * factor;
    int x1 = x0 + factor;
    int y1 = y0 + factor;

    x0 = max(0, min(x0, width - 1));
    y0 = max(0, min(y0, height - 1));
    x1 = max(0, min(x1, width - 1));
    y1 = max(0, min(y1, height - 1));

    int idx00 = idx2d(y0, x0, width);
    int idx10 = idx2d(y0, x1, width);
    int idx01 = idx2d(y1, x0, width);
    int idx11 = idx2d(y1, x1, width);

    bool has00 = mask[idx00];
    bool has10 = mask[idx10];
    bool has01 = mask[idx01];
    bool has11 = mask[idx11];

    float3 C00 = make_float3(0,0,0);
    float3 C10 = make_float3(0,0,0);
    float3 C01 = make_float3(0,0,0);
    float3 C11 = make_float3(0,0,0);

    if (has00) { uchar3 c=subsampled[idx00]; C00=make_float3(c.x,c.y,c.z); }
    if (has10) { uchar3 c=subsampled[idx10]; C10=make_float3(c.x,c.y,c.z); }
    if (has01) { uchar3 c=subsampled[idx01]; C01=make_float3(c.x,c.y,c.z); }
    if (has11) { uchar3 c=subsampled[idx11]; C11=make_float3(c.x,c.y,c.z); }

    if (!has00 && !has10 && !has01 && !has11) {
        int nx = max(0, min((x/factor)*factor, width-1));
        int ny = max(0, min((y/factor)*factor, height-1));
        return subsampled[idx2d(ny, nx, width)];
    }

    float dx = (float)(x - x0) / (float)(max(1, x1 - x0));
    float dy = (float)(y - y0) / (float)(max(1, y1 - y0));

    float w00 = (1.f - dx) * (1.f - dy);
    float w10 = dx        * (1.f - dy);
    float w01 = (1.f - dx) * dy;
    float w11 = dx        * dy;

    float3 out = make_float3(0,0,0);
    float wsum = 0.f;

    if (has00) { out.x += w00 * C00.x; out.y += w00*C00.y; out.z+=w00*C00.z; wsum+=w00; }
    if (has10) { out.x += w10 * C10.x; out.y += w10*C10.y; out.z+=w10*C10.z; wsum+=w10; }
    if (has01) { out.x += w01 * C01.x; out.y += w01*C01.y; out.z+=w01*C01.z; wsum+=w01; }
    if (has11) { out.x += w11 * C11.x; out.y += w11*C11.y; out.z+=w11*C11.z; wsum+=w11; }

    if (wsum > 0.f) {
        out.x /= wsum; out.y /= wsum; out.z /= wsum;
    }

    uchar3 outPix;
    outPix.x = (unsigned char)fminf(fmaxf(out.x, 0.f), 255.f);
    outPix.y = (unsigned char)fminf(fmaxf(out.y, 0.f), 255.f);
    outPix.z = (unsigned char)fminf(fmaxf(out.z, 0.f), 255.f);
    return outPix;
}

// Kernel: one block per tile, one thread per pixel in tile
__global__ void reconstructActiveTilesKernel(const uchar3* subsampled,
                                             const unsigned char* mask,
                                             uchar3* finalRecon,
                                             const unsigned char* tileActive,
                                             int width, int height,
                                             int tileSize,
                                             int tilesX, int tilesY,
                                             int factor)
{
    int tx = blockIdx.x;
    int ty = blockIdx.y;
    if (tx >= tilesX || ty >= tilesY) return;

    int tileIdx = ty * tilesX + tx;
    if (tileActive[tileIdx] == 0)
        return;     // static tile

    int lx = threadIdx.x;
    int ly = threadIdx.y;

    int x = tx * tileSize + lx;
    int y = ty * tileSize + ly;

    if (x >= width || y >= height) return;

    int idx = idx2d(y, x, width);

    if (mask[idx] == 1) {
        finalRecon[idx] = subsampled[idx];
    } else {
        finalRecon[idx] = bilinearPredictDevice(subsampled, mask,
                                                width, height,
                                                y, x, factor);
    }
}

void reconstructActiveTilesCUDA(const cv::Mat& subsampled,
                                const cv::Mat& mask,
                                const cv::Mat& prevRecon,
                                const cv::Mat& tileActive,
                                int tileSize,
                                int factor,
                                cv::Mat& finalRecon)
{
    if (subsampled.empty() || mask.empty() ||
        prevRecon.empty() || tileActive.empty())
        throw std::runtime_error("CUDA inputs empty");

    int width  = subsampled.cols;
    int height = subsampled.rows;

    int tilesY = tileActive.rows;
    int tilesX = tileActive.cols;

    finalRecon = prevRecon.clone();

    size_t numPix = (size_t)width * height;
    size_t frameBytes = numPix * sizeof(uchar3);
    size_t maskBytes  = numPix * sizeof(unsigned char);
    size_t tileBytes  = (size_t)tilesX * tilesY * sizeof(unsigned char);

    uchar3* d_sub = nullptr;
    unsigned char* d_mask = nullptr;
    uchar3* d_out = nullptr;
    unsigned char* d_tile = nullptr;

    cudaMalloc(&d_sub, frameBytes);
    cudaMalloc(&d_mask, maskBytes);
    cudaMalloc(&d_out, frameBytes);
    cudaMalloc(&d_tile, tileBytes);

    cudaMemcpy(d_sub,  subsampled.ptr<uchar3>(), frameBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask.ptr<unsigned char>(), maskBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out,  finalRecon.ptr<uchar3>(), frameBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tile, tileActive.ptr<unsigned char>(), tileBytes, cudaMemcpyHostToDevice);

    dim3 block(tileSize, tileSize);
    dim3 grid(tilesX, tilesY);

    reconstructActiveTilesKernel<<<grid, block>>>(d_sub, d_mask,
                                                  d_out, d_tile,
                                                  width, height,
                                                  tileSize,
                                                  tilesX, tilesY,
                                                  factor);
    cudaDeviceSynchronize();

    cudaMemcpy(finalRecon.ptr<uchar3>(), d_out, frameBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_sub);
    cudaFree(d_mask);
    cudaFree(d_out);
    cudaFree(d_tile);
}
