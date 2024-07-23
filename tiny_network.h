#ifndef TINY_NETWORK_H
#define TINY_NETWORK_H

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layer_sizes);
    std::vector<float> forward(const std::vector<float>& input);
    void backward(const std::vector<float>& input, const std::vector<float>& target, float learning_rate);

private:
    std::vector<int> layer_sizes;
    std::vector<std::vector<std::vector<float>>> weights; // Weights for each layer
    std::vector<std::vector<float>> biases; // Biases for each layer
    std::vector<std::vector<float>> activations; // Activations for each layer
    std::vector<std::vector<float>> z_values; // Linear combinations for each layer
};

#endif // TINY_NETWORK_H

