#include "recon.h"
#include <algorithm>

void subsampleFrame(const cv::Mat& input,
                    cv::Mat& subsampled,
                    cv::Mat& mask,
                    int factor)
{
    subsampled = cv::Mat::zeros(input.size(), input.type());
    mask       = cv::Mat::zeros(input.size(), CV_8UC1);

    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) {
            if ((x % factor == 0) && (y % factor == 0)) {
                subsampled.at<cv::Vec3b>(y, x) = input.at<cv::Vec3b>(y, x);
                mask.at<uint8_t>(y, x) = 1;
            }
        }
    }
}

#include <algorithm> // for std::clamp

static cv::Vec3b bilinearPredict(const cv::Mat& subsampled,
                                 const cv::Mat& mask,
                                 int y, int x,
                                 int factor)
{
    int rows = subsampled.rows;
    int cols = subsampled.cols;

    // Find top-left grid corner
    int x0 = (x / factor) * factor;
    int y0 = (y / factor) * factor;

    int x1 = x0 + factor;
    int y1 = y0 + factor;

    // Clamp to image bounds
    x0 = std::max(0, std::min(x0, cols - 1));
    y0 = std::max(0, std::min(y0, rows - 1));
    x1 = std::max(0, std::min(x1, cols - 1));
    y1 = std::max(0, std::min(y1, rows - 1));

    // Handle degenerate case near borders
    if (x0 == x1 && y0 == y1 && mask.at<uint8_t>(y0, x0) == 1) {
        return subsampled.at<cv::Vec3b>(y0, x0);
    }

    // Helper: get color if sampled, else mark as missing
    auto getColorIfSampled = [&](int yy, int xx, bool &has) -> cv::Vec3d {
        if (mask.at<uint8_t>(yy, xx) == 1) {
            has = true;
            cv::Vec3b c = subsampled.at<cv::Vec3b>(yy, xx);
            return cv::Vec3d(c[0], c[1], c[2]);
        }
        has = false;
        return cv::Vec3d(0.0, 0.0, 0.0);
    };

    bool has00, has10, has01, has11;
    cv::Vec3d C00 = getColorIfSampled(y0, x0, has00);
    cv::Vec3d C10 = getColorIfSampled(y0, x1, has10);
    cv::Vec3d C01 = getColorIfSampled(y1, x0, has01);
    cv::Vec3d C11 = getColorIfSampled(y1, x1, has11);

    // If we somehow have no valid corners, just fall back to nearest sampled grid point
    if (!has00 && !has10 && !has01 && !has11) {
        int nx = (x / factor) * factor;
        int ny = (y / factor) * factor;
        nx = std::max(0, std::min(nx, cols - 1));
        ny = std::max(0, std::min(ny, rows - 1));
        return subsampled.at<cv::Vec3b>(ny, nx);
    }

    double dx = (x1 == x0) ? 0.0 : double(x - x0) / double(x1 - x0);
    double dy = (y1 == y0) ? 0.0 : double(y - y0) / double(y1 - y0);

    // Standard bilinear weights
    double w00 = (1.0 - dx) * (1.0 - dy);
    double w10 = dx * (1.0 - dy);
    double w01 = (1.0 - dx) * dy;
    double w11 = dx * dy;

    cv::Vec3d color(0.0, 0.0, 0.0);
    double wSum = 0.0;

    if (has00) { color += w00 * C00; wSum += w00; }
    if (has10) { color += w10 * C10; wSum += w10; }
    if (has01) { color += w01 * C01; wSum += w01; }
    if (has11) { color += w11 * C11; wSum += w11; }

    if (wSum > 0.0) {
        color /= wSum;  // renormalize if some corners were missing
    }

    cv::Vec3b out;
    out[0] = static_cast<uchar>(std::clamp(color[0], 0.0, 255.0));
    out[1] = static_cast<uchar>(std::clamp(color[1], 0.0, 255.0));
    out[2] = static_cast<uchar>(std::clamp(color[2], 0.0, 255.0));
    return out;
}

void reconstructNaive(const cv::Mat& subsampled,
                      const cv::Mat& mask,
                      int factor,
                      cv::Mat& output)
{
    output = subsampled.clone();

    for (int y = 0; y < subsampled.rows; ++y) {
        for (int x = 0; x < subsampled.cols; ++x) {
            if (mask.at<uint8_t>(y, x) == 0) { // missing pixel
                output.at<cv::Vec3b>(y, x) =
                    bilinearPredict(subsampled, mask, y, x, factor);
            }
        }
    }
}

// Small 3x3 averaging stencil used during refinement
static cv::Vec3b average3x3MissingAware(const cv::Mat& src,
                                        const cv::Mat& mask,
                                        int y, int x)
{
    int rows = src.rows;
    int cols = src.cols;

    int count = 0;
    int sumB = 0, sumG = 0, sumR = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        int ny = y + dy;
        if (ny < 0 || ny >= rows) continue;

        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            if (nx < 0 || nx >= cols) continue;

            // We use all neighbors (both known + previously refined),
            // but we'll keep known pixels pinned in the outer loop.
            cv::Vec3b v = src.at<cv::Vec3b>(ny, nx);
            sumB += v[0];
            sumG += v[1];
            sumR += v[2];
            count++;
        }
    }

    if (count == 0) {
        return src.at<cv::Vec3b>(y, x);
    }

    return cv::Vec3b(sumB / count, sumG / count, sumR / count);
}

void iterativeRefine(cv::Mat& img,
                     const cv::Mat& mask,
                     int iterations)
{
    CV_Assert(img.size() == mask.size());
    CV_Assert(mask.type() == CV_8UC1);
    CV_Assert(img.type() == CV_8UC3);

    cv::Mat cur  = img.clone();
    cv::Mat next = img.clone();

    for (int it = 0; it < iterations; ++it) {
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                if (mask.at<uint8_t>(y, x) == 1) {
                    // Known pixels stay fixed – they “anchor” the solution
                    next.at<cv::Vec3b>(y, x) = cur.at<cv::Vec3b>(y, x);
                } else {
                    // Only refine missing pixels
                    next.at<cv::Vec3b>(y, x) =
                        average3x3MissingAware(cur, mask, y, x);
                }
            }
        }
        // Jacobi-style update: swap buffers
        std::swap(cur, next);
    }

    img = cur; // Final refined image
}
