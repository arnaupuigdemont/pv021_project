#include "neuron.h"

class Neuron {
public:
    std::vector<double> weights;  // Neuron weights
    double bias;                  // Neuron bias

    Neuron(int inputs) {
        for (int i = 0; i < inputs; ++i) {
            weights.push_back(random_weight());
        }
        bias = random_weight();
    }

    double activate(const std::vector<double>& inputs) {
        double activation = bias;
        for (int i = 0; i < inputs.size(); ++i) {
            activation += weights[i] * inputs[i];
        }
        return sigmoid(activation);
    }
};
