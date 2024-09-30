#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include "neural_net.h"  

// Function to read CSV data
std::vector<std::vector<double>> read_csv(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<std::vector<double>> data;

    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }
        data.push_back(row);
    }

    return data;
}

// Function to load labels from a CSV file
std::vector<int> load_labels(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<int> labels;

    while (std::getline(file, line)) {
        labels.push_back(std::stoi(line));
    }

    return labels;
}

// Function to write predictions to a CSV file
void write_predictions(const std::string& filename, const std::vector<int>& predictions) {
    std::ofstream file(filename);
    for (const int& pred : predictions) {
        file << pred << "\n";
    }
}

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
