#include "layer.h"

class Layer {
public:
    std::vector<Neuron> neurons;

    Layer(int num_neurons, int num_inputs) {
        for (int i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(num_inputs);
        }
    }

    std::vector<double> feedForward(const std::vector<double>& inputs) {
        std::vector<double> outputs;
        for (auto& neuron : neurons) {
            outputs.push_back(neuron.activate(inputs));
        }
        return outputs;
    }
};
