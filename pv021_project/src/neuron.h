#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron {
public:
    std::vector<double> weights;
    double bias;

    // Constructor
    Neuron(int inputs);

    // Activation function: sigmoid
    double activate(const std::vector<double>& inputs);
};

#endif // NEURON_H
