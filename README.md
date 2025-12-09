# Parallel Video Reconstruction via Motion-Aware Predictive Upsampling
*A 15-418 / 15-618 Final Project — Fall 2025*
**Authors:** Andrew Kim & Martin Lee

> Reconstructing high-resolution video from aggressively subsampled frames using spatial interpolation, iterative refinement, and motion-aware selective computation. Pipeline implemented in C++ (serial + OpenMP) and CUDA, with detailed metrics (timing, tile activation, MSE/PSNR/SSIM).

Website Link: https://martinlhw.github.io/PVR-via-MAPU/

## 1. Overview
Modern video pipelines—mobile cameras, cloud gaming, streaming codecs—are increasingly **bandwidth-bound**. A common mitigation strategy is to **subsample** each frame and then use compute-heavy prediction to restore missing pixels.

This project builds a full **parallel video reconstruction system** that:
- Subsamples frames by configurable factors
- Interpolates missing pixels spatially
- Performs iterative 3×3 stencil refinement
- Uses motion-aware tile classification for selective refinement
- Parallelizes via **OpenMP** and **CUDA**
- Produces detailed **metrics** for quality and performance

## 2. Pipeline Architecture
### 2.1 Subsampling & Masking
We retain 1 of N pixels (N = subsampleFactor²), creating a sparse frame + mask.

### 2.2 Spatial Interpolation
Bilinear interpolation provides an initial smooth estimate.

### 2.3 Iterative Refinement (Stencil)
Multiple Jacobi 3×3 smoothing iterations propagate known information.

### 2.4 Motion-Aware Tile Classification
Each frame is partitioned into TILE_SIZE × TILE_SIZE tiles.
Using **Sum of Absolute Differences (SAD)** between consecutive subsampled frames:
- **Static tiles** reuse previous reconstruction
- **Active tiles** undergo full refinement

This drastically reduces computation on low-motion scenes.

## 3. Parallelization Strategy
### Serial Baseline
Reference CPU-only implementation for correctness & metrics.

### OpenMP CPU Path
Parallelizes subsampling, interpolation, and refinement loops.

### CUDA GPU Path
Active tiles are processed using:
- Shared memory tiling  
- Coalesced memory access  
- Multi-iteration refinement loops  

### Irregular Parallelism via Worklists
Dynamic active-tile sets create real-world variability in workload.

## 4. Quality Metrics
Computed automatically:
- MSE
- PSNR
- SSIM

Temporal blending was intentionally removed due to motion-smearing artifacts.

## 5. Build Instructions
### CPU-only
```
mkdir build && cd build
cmake ..
make -j
```

### CUDA-enabled
```
mkdir build && cd build
cmake -DUSE_CUDA_REFINEMENT=ON ..
make -j
```

## 6. Running the Pipeline
General format:
```
./serial <input> <output or -> [subsample tileSize sadThresh iterations] [flags]
```

### Useful Flags
| Flag | Description |
|------|-------------|
| `--force-images` | Interpret input as directory of frames |
| `-` | Disable video output |
| `--no-metrics` | Disable metric collection |
| `--csv <file>` | Log metrics to CSV |

### Default-Knobs CUDA Run (with metrics ON)
```
./serial frames_sample - --force-images
```

Defaults:
- subsampleFactor = 2  
- tileSize = 2  
- sadThresh = 1.0  
- iterations = 3  
- metrics enabled  
- CUDA mode auto-selected if available

## 7. Parameter Knobs
| Parameter | Meaning | Range |
|----------|----------|--------|
| subsampleFactor | pixel drop factor | 2–8 |
| tileSize | tile classification size | 2,4,8 |
| sadThresh | motion threshold | 0.5–3.0 |
| iterations | refinement passes | 1–5 |

## 8. Performance Evaluation
We measure:
- CPU OpenMP scaling
- GPU throughput vs CPU
- Active tile ratios over time
- Stage-level timing (subsample, classify, refine)
- PSNR/SSIM vs knob settings

GPU refinement + motion-aware tiles yields large speedups.

## 9. Directory Structure
```
├── serial.cpp
├── recon.cpp / recon.h
├── recon_cuda.cu / recon_cuda.h
├── metrics.cpp / metrics.h
├── frames_sample/
└── sweep scripts
```

## 10. Running on PSC Bridges-2
Example workflow:
```
interact -gpu -gpus-per-node=1 -t 02:00:00
./serial frames_60s - --force-images
./serial frames_60s - 2 2 1.0 3 --force-images --csv results.csv
```

## 11. Acknowledgements
Developed for CMU **15‑418/15‑618: Parallel Computer Architecture & Programming (Fall 2025)**.
