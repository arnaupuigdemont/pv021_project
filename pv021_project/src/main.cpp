#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include "neural_net.h"  
#include "aux_func.h"

int main() {
    // Load training data
    auto train_vectors = read_csv("data/fashion_mnist_train_vectors.csv");
    auto train_labels = load_labels("data/fashion_mnist_train_labels.csv");

    // Load test data
    auto test_vectors = read_csv("data/fashion_mnist_test_vectors.csv");
    auto test_labels = load_labels("data/fashion_mnist_test_labels.csv");

    //min max normalization (to implement)

    // Initialize the neural network
    NeuralNetwork nn(784, 64, 10);  // 784 input nodes (28x28), 128 hidden neurons, 10 output nodes

    // Train the neural network
    int epochs = 10;  // Number of epochs
    double learning_rate = 0.01;  // Learning rate
    nn.train(train_vectors, train_labels, epochs, learning_rate);

    // Make predictions on training set
    std::vector<int> train_predictions;
    for (const auto& input : train_vectors) {
        auto output = nn.feedforward(input);
        train_predictions.push_back(std::distance(output.begin(), std::max_element(output.begin(), output.end())));
    }

    // Write training predictions to CSV
    write_predictions("train_predictions.csv", train_predictions);

    // Make predictions on test set
    std::vector<int> test_predictions;
    for (const auto& input : test_vectors) {
        auto output = nn.feedforward(input);
        test_predictions.push_back(std::distance(output.begin(), std::max_element(output.begin(), output.end())));
    }

    // Write test predictions to CSV
    write_predictions("test_predictions.csv", test_predictions);

    std::cout << "Training and testing completed. Predictions saved." << std::endl;

    return 0;
}
