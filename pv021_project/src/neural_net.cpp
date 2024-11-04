#include "neural_net.h"

class Neural_net {
public:
    std::vector<Layer> layers;

    Neural_net(int num_inputs, int num_hidden, int num_outputs) {
        // Hidden layer
        layers.emplace_back(num_hidden, num_inputs);
        // Output layer
        layers.emplace_back(num_outputs, num_hidden);
    }

    std::vector<double> feedForward(const std::vector<double>& inputs) {
        std::vector<double> activations = inputs;
        for (auto& layer : layers) {
            activations = layer.feedForward(activations);
        }
        return activations;
    }

    // Backpropagation (to be implemented later)
    void backPropagation(const std::vector<double>& inputs, const std::vector<double>& target, double learning_rate) {
        // Here, implement backpropagation, adjusting weights and biases
    }
};
