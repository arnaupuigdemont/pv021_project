#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "neuron.h" // Include the Neuron class

class Layer {
public:
    std::vector<Neuron> neurons;

    // Constructor
    Layer(int num_neurons, int num_inputs);

    // Feedforward: passes inputs through the layer
    std::vector<double> feedforward(const std::vector<double>& inputs);
};

#endif // LAYER_H
