#include "TinyNn_optimized.h"
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h> // Include OpenMP header

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes) : layer_sizes(layer_sizes) {
    // Initialize weights and biases randomly
    for (size_t i = 1; i < layer_sizes.size(); ++i) {
        int rows = layer_sizes[i];
        int cols = layer_sizes[i - 1];
        std::vector<std::vector<float>> layer_weights(rows, std::vector<float>(cols));
        std::vector<float> layer_biases(rows);
        // Initialize weights with small random values and biases with zeros
        for (int j = 0; j < rows; ++j) {
            std::generate(layer_weights[j].begin(), layer_weights[j].end(), []() { return static_cast<float>(rand()) / RAND_MAX - 0.5f; });
            layer_biases[j] = 0.0f;
        }
        weights.push_back(layer_weights);
        biases.push_back(layer_biases);
    }
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) {
    activations.clear();
    z_values.clear();
    activations.push_back(input);

    std::vector<float> activation = input;
    for (size_t i = 0; i < weights.size(); ++i) {
        std::vector<float> z(layer_sizes[i + 1], 0.0f);

        #pragma omp parallel for // Parallelize the outer loop
        for (size_t j = 0; j < layer_sizes[i + 1]; ++j) {
            for (size_t k = 0; k < layer_sizes[i]; ++k) {
                z[j] += weights[i][j][k] * activation[k];
            }
            z[j] += biases[i][j];
            z[j] = 1.0 / (1.0 + std::exp(-z[j])); // Sigmoid activation
        }
        z_values.push_back(z);
        activation = z;
        activations.push_back(activation);
    }
    return activation;
}

void NeuralNetwork::backward(const std::vector<float>& input, const std::vector<float>& target, float learning_rate) {
    std::vector<float> output_gradients = activations.back();
    for (size_t i = 0; i < output_gradients.size(); ++i) {
        output_gradients[i] -= target[i];
    }

    std::vector<std::vector<float>> hidden_gradients(weights.size());
    for (int l = weights.size() - 1; l >= 0; --l) {
        hidden_gradients[l].resize(layer_sizes[l + 1], 0.0f);

        #pragma omp parallel for // Parallelize the outer loop
        for (size_t j = 0; j < layer_sizes[l + 1]; ++j) {
            float gradient = output_gradients[j] * activations[l + 1][j] * (1.0f - activations[l + 1][j]);
            hidden_gradients[l][j] = gradient;
            for (size_t k = 0; k < layer_sizes[l]; ++k) {
                #pragma omp atomic // Ensure atomic operation for weight update
                weights[l][j][k] -= learning_rate * gradient * activations[l][k];
            }
            #pragma omp atomic // Ensure atomic operation for bias update
            biases[l][j] -= learning_rate * gradient;
        }

        if (l > 0) {
            std::vector<float> next_output_gradients(layer_sizes[l], 0.0f);

            #pragma omp parallel for // Parallelize the outer loop
            for (size_t k = 0; k < layer_sizes[l]; ++k) {
                for (size_t j = 0; j < layer_sizes[l + 1]; ++j) {
                    next_output_gradients[k] += hidden_gradients[l][j] * weights[l][j][k];
                }
            }
            output_gradients = next_output_gradients;
        }
    }
}

