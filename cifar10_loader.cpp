#include "cifar10_loader.h"
#include <stdio.h>
#include <stdlib.h>

std::vector<CIFAR10Image> load_cifar10_bin(const char* data_path) {
    std::vector<CIFAR10Image> dataset(NUM_IMAGES);
    FILE* file = fopen(data_path, "rb");

    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", data_path);
        exit(1);
    }

    for (int i = 0; i < NUM_IMAGES; ++i) {
        unsigned char label;
        fread(&label, LABEL_BYTES, 1, file);

        unsigned char buffer[IMAGE_BYTES];
        fread(buffer, IMAGE_BYTES, 1, file);

        cv::Mat img(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3);

        for (int row = 0; row < IMAGE_SIZE; ++row) {
            for (int col = 0; col < IMAGE_SIZE; ++col) {
                img.at<cv::Vec3b>(row, col)[0] = buffer[row * IMAGE_SIZE + col]; // Blue
                img.at<cv::Vec3b>(row, col)[1] = buffer[IMAGE_SIZE * IMAGE_SIZE + row * IMAGE_SIZE + col]; // Green
                img.at<cv::Vec3b>(row, col)[2] = buffer[2 * IMAGE_SIZE * IMAGE_SIZE + row * IMAGE_SIZE + col]; // Red
            }
        }

        dataset[i].image = img;
        dataset[i].label = (int)label;
    }

    fclose(file);
    return dataset;
}

