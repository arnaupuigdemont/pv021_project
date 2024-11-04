#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include "layer.h" // Include the Layer class

class Neural_net {
public:
    std::vector<Layer> layers;

    //constructor
    Neural_net(int num_inputs, int num_hidden, int num_outputs);

    // Feedforward: passes inputs through the network
    std::vector<double> feedForward(const std::vector<double>& inputs);

    // Backpropagation: adjusts weights and biases based on the error
    void backPropagation(const std::vector<double>& inputs, const std::vector<double>& target, double learning_rate);

    // Train the network
    void train(const std::vector<std::vector<double>>& inputs, const std::vector<int>& labels, int epochs, double learning_rate);
};

#endif // NEURAL_NETWORK_H
