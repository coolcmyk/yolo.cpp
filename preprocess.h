#ifndef PREPROCESS_H
#define PREPROCESS_H


#include <vector>
#include <opencv2/opencv.hpp>


// struct to store preprocessed images
cv::Mat preprocess(const cv::Mat& image) {
    cv::Mat img;
    image.convertTo(img, CV_32F); // Convert to float
    img = img / 255.0; // Normalize to [0, 1]
    return img;
}

#endif // PREPROCESS_H
