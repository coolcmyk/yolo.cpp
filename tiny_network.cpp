#include "tiny_network.h"
#include <random>
#include <cmath>
#include <algorithm>

//NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes) : layer_sizes(layer_sizes) {
//    initialize_weights();
//}

//void NeuralNetwork::initialize_weights() {
//    weights.clear();
//    biases.clear();
//    activations.clear();
//     z_values.clear();

//    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
//        weights.push_back(random_matrix(layer_sizes[i + 1], layer_sizes[i]));
//        biases.push_back(std::vector<float>(layer_sizes[i + 1], 0.0f));
//        activations.push_back(std::vector<float>(layer_sizes[i + 1], 0.0f));
//        z_values.push_back(std::vector<float>(layer_sizes[i + 1], 0.0f));
//    }
//}

//std::vector<std::vector<float>> NeuralNetwork::random_matrix(int rows, int cols) {
//    std::vector<std::vector<float>> mat(rows, std::vector<float>(cols));
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    std::uniform_real_distribution<> dis(-0.1, 0.1);
//    for (int i = 0; i < rows; ++i) {
//        for (int j = 0; j < cols; ++j) {
//            mat[i][j] = dis(gen);
//        }
//    }
//    return mat;
//}

//std::vector<float> NeuralNetwork::mat_vec_mult(const std::vector<std::vector<float>>& mat, const std::vector<float>& vec) {
//    std::vector<float> result(mat.size(), 0.0);
//    for (size_t i = 0; i < mat.size(); ++i) {
//        for (size_t j = 0; j < mat[i].size(); ++j) {
//            result[i] += mat[i][j] * vec[j];
//        }
//    }
//    return result;
//}

//void NeuralNetwork::apply_activation(std::vector<float>& vec) {
//    for (float& val : vec) {
//        val = 1.0 / (1.0 + std::exp(-val)); // Sigmoid activation
//    }
//}

//void NeuralNetwork::apply_derivative(std::vector<float>& vec) {
//    for (float& val : vec) {
//        val *= (1.0 - val); // Derivative of sigmoid
//    }
//}

//std::vector<float> NeuralNetwork::compute_output_gradients(const std::vector<float>& target) {
//    std::vector<float> gradients(layer_sizes.back());
//    for (size_t i = 0; i < layer_sizes.back(); ++i) {
//        gradients[i] = (activations.back()[i] - target[i]);
//    }
//    apply_derivative(gradients);
//    return gradients;
//}

//std::vector<std::vector<float>> NeuralNetwork::compute_hidden_gradients(const std::vector<float>& output_gradients) {
//    std::vector<std::vector<float>> hidden_gradients(layer_sizes.size() - 1);
//    std::vector<float> next_layer_gradients = output_gradients;

//    for (int i = layer_sizes.size() - 2; i >= 0; --i) {
//        std::vector<float> gradients(layer_sizes[i], 0.0);
//        for (size_t j = 0; j < layer_sizes[i]; ++j) {
//            float sum = 0.0;
//            for (size_t k = 0; k < layer_sizes[i + 1]; ++k) {
//                sum += next_layer_gradients[k] * weights[i + 1][k][j];
//            }
//            gradients[j] = sum;
//        }
//        apply_derivative(gradients);
//        hidden_gradients[i] = gradients;
//        next_layer_gradients = gradients;
//    }
//
//    return hidden_gradients;
//}

//void NeuralNetwork::update_weights_biases(const std::vector<float>& input, const std::vector<float>& hidden_gradients,
//                                          const std::vector<float>& output_gradients, float learning_rate) {
    // Update weights and biases for output layer
//    for (size_t i = 0; i < layer_sizes.back(); ++i) {
//        for (size_t j = 0; j < layer_sizes[layer_sizes.size() - 2]; ++j) {
//            weights.back()[i][j] -= learning_rate * output_gradients[i] * activations[layer_sizes.size() - 2][j];
//      }
//        biases.back()[i] -= learning_rate * output_gradients[i];
//    }

    // Update weights and biases for hidden layers
  // std::vector<float> previous_activations = input;
    //for (int i = layer_sizes.size() - 2; i >= 0; --i) {
     //   for (size_t j = 0; j < layer_sizes[i + 1]; ++j) {
       //     for (size_t k = 0; k < layer_sizes[i]; ++k) {
         //       weights[i][j][k] -= learning_rate * hidden_gradients[i][j] * previous_activations[k];
           // }
           // biases[i][j] -= learning_rate * hidden_gradients[i][j];
  //      }
    //    if (i > 0) {
      //      previous_activations = activations[i - 1];
        //}
//    }
//}

//std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) {
  //  activations[0] = input;
    //z_values[0] = mat_vec_mult(weights[0], input);

    //for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
      //  apply_activation(z_values[i]);
        //activations[i + 1] = z_values[i];
        //if (i < layer_sizes.size() - 2) {
          //  z_values[i + 1] = mat_vec_mult(weights[i + 1], activations[i + 1]);
     //   }
    //}

    //return activations.back();
//}

//void NeuralNetwork::backward(const std::vector<float>& input, const std::vector<float>& target, float learning_rate) {
  //  std::vector<float> output_gradients = compute_output_gradients(target);
    //std::vector<std::vector<float>> hidden_gradients = compute_hidden_gradients(output_gradients);

//    update_weights_biases(input, hidden_gradients, output_gradients, learning_rate);
//
#include "tiny_network.h"
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>

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
        for (size_t j = 0; j < layer_sizes[l + 1]; ++j) {
            float gradient = output_gradients[j] * activations[l + 1][j] * (1.0f - activations[l + 1][j]);
            hidden_gradients[l][j] = gradient;
            for (size_t k = 0; k < layer_sizes[l]; ++k) {
                weights[l][j][k] -= learning_rate * gradient * activations[l][k];
            }
            biases[l][j] -= learning_rate * gradient;
        }
        if (l > 0) {
            std::vector<float> next_output_gradients(layer_sizes[l], 0.0f);
            for (size_t k = 0; k < layer_sizes[l]; ++k) {
                for (size_t j = 0; j < layer_sizes[l + 1]; ++j) {
                    next_output_gradients[k] += hidden_gradients[l][j] * weights[l][j][k];
                }
            }
            output_gradients = next_output_gradients;
        }
    }
}

