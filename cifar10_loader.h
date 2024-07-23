#ifndef CIFAR10_LOADER_H
#define CIFAR10_LOADER_H

#include <opencv2/opencv.hpp>
#include <vector>

// Constants
#define IMAGE_SIZE 32
#define NUM_CHANNELS 3
#define NUM_IMAGES 10000
#define IMAGE_BYTES (IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS)
#define LABEL_BYTES 1
#define RECORD_BYTES (IMAGE_BYTES + LABEL_BYTES)

// Struct to store CIFAR-10 image and label
typedef struct {
    cv::Mat image;
    int label;
} CIFAR10Image;

// Function declarations
std::vector<CIFAR10Image> load_cifar10_bin(const char* data_path);

#endif // CIFAR10_LOADER_H

