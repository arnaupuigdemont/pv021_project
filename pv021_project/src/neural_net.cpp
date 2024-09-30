class NeuralNet {
public:
    std::vector<Layer> layers;

    NeuralNetwork(int num_inputs, int num_hidden, int num_outputs) {
        // Hidden layer
        layers.emplace_back(num_hidden, num_inputs);
        // Output layer
        layers.emplace_back(num_outputs, num_hidden);
    }

    std::vector<double> feedforward(const std::vector<double>& inputs) {
        std::vector<double> activations = inputs;
        for (auto& layer : layers) {
            activations = layer.feedforward(activations);
        }
        return activations;
    }

    // Backpropagation (to be implemented later)
    void backpropagation(const std::vector<double>& inputs, const std::vector<double>& target, double learning_rate) {
        // Here, implement backpropagation, adjusting weights and biases
    }
};
