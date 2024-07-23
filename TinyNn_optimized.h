#ifndef TINYNN_OPTIMIZED_H
#define TINYNN_OPTIMIZED_H

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layer_sizes);
    std::vector<float> forward(const std::vector<float>& input);
    void backward(const std::vector<float>& input, const std::vector<float>& target, float learning_rate);

private:
    std::vector<int> layer_sizes;
    std::vector<std::vector<std::vector<float>>> weights;
    std::vector<std::vector<float>> biases;
    std::vector<std::vector<float>> activations;
    std::vector<std::vector<float>> z_values;
};

#endif // TINYNN_OPTIMIZED_H

