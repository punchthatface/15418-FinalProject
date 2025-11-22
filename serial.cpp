#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_video> <output_video>\n";
        return 1;
    }

    std::string inputPath  = argv[1];
    std::string outputPath = argv[2];

    cv::VideoCapture cap(inputPath);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open input video: " << inputPath << "\n";
        return 1;
    }

    int width  = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    printf("%f",fps);
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
    int frameCount = 0;

    while (true) {
        if (!cap.read(frame)) break;  // end of video

        // For now, no processing — just write the frame back out:
        writer.write(frame);
        frameCount++;
    }

    std::cout << "Processed " << frameCount << " frames.\n";
    return 0;
}
