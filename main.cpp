#include <opencv2/opencv.hpp>
#include "cifar10_loader.h"
#include "TinyNn_optimized.h"
#include "preprocess.h"
#define NUM_DATA 1
#define TEST_PATH "dataset/test_batch.bin"
#define LEARNING_RATE 0.5
#define NUM_EPOCHS 1000
#define BATCH_SIZE 64 

int main() {
    //loading the dataset, boring stuffs
    std::vector<CIFAR10Image> combined_dataset;
    for(int i = 0; i < NUM_DATA; i++){
        std::string data_path = "dataset/data_batch_" + std::to_string(i + 1) + ".bin";
        
        // Example of using the data path
        std::cout << "Processing file: " << data_path << std::endl;

        std::vector<CIFAR10Image> dataset = load_cifar10_bin(data_path.c_str());

        // Append the loaded dataset to the combined_dataset vector
        if (!dataset.empty()) {
            combined_dataset.insert(combined_dataset.end(), dataset.begin(), dataset.end());
        } else {
            std::cerr << "Failed to load dataset from " << data_path << std::endl;
        }


    }
    std::cout << "Total number of images in the combined dataset: " << combined_dataset.size() << std::endl;
   

    //function demo
    //const char* data_path = "dataset/data_batch_1.bin";
    //std::vector<CIFAR10Image> dataset = load_cifar10_bin(data_path);

    //if (!combined_dataset.empty()) {
        
    //    cv::imshow("CIFAR-10 Image", combined_dataset[0].image);
    //    printf("Label: %d\n", combined_dataset[0].label);
    //    cv::waitKey(0);
    //}
   


    //pre-cooking || preprocess the dataset
    std::vector<cv::Mat> images;
    std::vector<int> labels;
    for (const auto& item : combined_dataset) {
        images.push_back(preprocess(item.image));
        labels.push_back(item.label);
    }
   
    // Split into train and validation sets (80% train, 20% validation)
    int num_train = static_cast<int>(0.8 * images.size());
    std::vector<cv::Mat> train_images(images.begin(), images.begin() + num_train);
    std::vector<int> train_labels(labels.begin(), labels.begin() + num_train);
    std::vector<cv::Mat> val_images(images.begin() + num_train, images.end());
    std::vector<int> val_labels(labels.begin() + num_train, labels.end());



    //cooking stuffs
    //each numbers in layer_sizes in order is input, first hidden layer, second hidden layer and finally an
    //output layer, the reasoning in each numbers is:
    //3072 => because of 32 x 32 x 3 after flattening of cifar10 dataset, 
    //128 is just a wild guess for first hidden layers (?) or some reasoning of math that
    //myself couldnt understand yet, goes the same for the second hidden layers, 
    //and finally the output layers, 10 is for representative of confidence in each classes
    //so if i used cifar100 it would be 100 instead of 10, but i guess the hidden layer cant be less
    //than the output layer so there were supposed to be an adjustment if i were using a different
    //datasets
    std::vector<int> layer_sizes = {3072, 128, 64, 10}; //input -> hidden1 -> hidden2 -> output
    NeuralNetwork nn(layer_sizes);



    // Training loop
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        float epoch_loss = 0.0;
        int num_correct = 0;

        for (size_t start = 0; start < combined_dataset.size(); start += BATCH_SIZE) {
            size_t end = std::min(start + BATCH_SIZE, combined_dataset.size());

            std::vector<std::vector<float>> batch_inputs;
            std::vector<std::vector<float>> batch_targets;

            for (size_t i = start; i < end; ++i) {
                
                const auto& item = combined_dataset[i];
                
                std::vector<float> input(item.image.total());
                std::memcpy(input.data(), item.image.ptr<float>(), item.image.total() * sizeof(float));
                batch_inputs.push_back(input);
                
                std::vector<float> target(layer_sizes.back(), 0.0f);
                target[item.label] = 1.0f; // One-hot encoding
                batch_targets.push_back(target);
            }

            for (size_t i = 0; i < batch_inputs.size(); ++i) {
                std::vector<float> output = nn.forward(batch_inputs[i]);
                nn.backward(batch_inputs[i], batch_targets[i], LEARNING_RATE);

                int predicted_label = std::max_element(output.begin(), output.end()) - output.begin();
                if (predicted_label == std::distance(batch_targets[i].begin(), std::max_element(batch_targets[i].begin(), batch_targets[i].end()))) {
                    num_correct++;
                }
            }
        }

        std::cout << "Epoch " << epoch << ": Accuracy = " << (num_correct / static_cast<float>(combined_dataset.size())) << std::endl;
    }

    // Loading and testing on the test set
    std::vector<CIFAR10Image> testset = load_cifar10_bin(TEST_PATH);
    std::vector<cv::Mat> test_images;
    std::vector<int> test_labels;
    for (const auto& item : testset) {
        test_images.push_back(preprocess(item.image));
        test_labels.push_back(item.label);
    }

    int num_correct_test = 0;
    for (size_t i = 0; i < test_images.size(); ++i) {
        const auto& item = combined_dataset[i];
        std::vector<float> input(item.image.total());
        std::memcpy(input.data(), item.image.ptr<float>(), item.image.total() * sizeof(float));

        std::vector<float> output = nn.forward(input);

        int predicted_label = std::max_element(output.begin(), output.end()) - output.begin();
        if (predicted_label == test_labels[i]) {
            num_correct_test++;
        }
    }
    std::cout << "Test Accuracy = " << (num_correct_test / static_cast<float>(test_images.size())) << std::endl;

    return 0;
}

