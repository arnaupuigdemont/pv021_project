#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include "neural_net.h"  
#include "aux_func.h"

using namespace std;

int main() {
    // Load training data
    auto train_vectors = read_csv("data/fashion_mnist_train_vectors.csv");
    auto train_labels = load_labels("data/fashion_mnist_train_labels.csv");

    // Load test data
    auto test_vectors = read_csv("data/fashion_mnist_test_vectors.csv");
    auto test_labels = load_labels("data/fashion_mnist_test_labels.csv");

    //min max normalization (to implement)

    //validation split (to implement)

    //hiddden and output vectors
    vector<double> hidden1(HIDDEN_SIZE1);
    vector<double> hidden2(HIDDEN_SIZE2);
    vector<double> output(OUTPUT_SIZE);

    // weights and biases
    random_device rd;
    mt19937 gen(42);
    uniform_real_distribution<> dis(-0.1, 0.1);
    vector<double> hidden_weights1(INPUT_SIZE * HIDDEN_SIZE1);
    vector<double> hidden_weights2(HIDDEN_SIZE1 * HIDDEN_SIZE2);
    vector<double> output_weights(HIDDEN_SIZE2 * OUTPUT_SIZE);
    vector<double> hidden_bias1(HIDDEN_SIZE1);
    vector<double> hidden_bias2(HIDDEN_SIZE2);
    vector<double> output_bias(OUTPUT_SIZE);
    for (auto &w : hidden_weights1) w = dis(gen);
    for (auto &w : hidden_weights2) w = dis(gen);
    for (auto &w : output_weights) w = dis(gen);
    for (auto &b : hidden_bias1) b = dis(gen);
    for (auto &b : hidden_bias2) b = dis(gen);
    for (auto &b : output_bias) b = dis(gen);

    // adam weights
    vector<double> m_hidden_weights1(INPUT_SIZE * HIDDEN_SIZE1, 0.0);
    vector<double> v_hidden_weights1(INPUT_SIZE * HIDDEN_SIZE1, 0.0);
    vector<double> m_hidden_weights2(HIDDEN_SIZE1 * HIDDEN_SIZE2, 0.0);
    vector<double> v_hidden_weights2(HIDDEN_SIZE1 * HIDDEN_SIZE2, 0.0);
    vector<double> m_output_weights(HIDDEN_SIZE2 * OUTPUT_SIZE, 0.0);
    vector<double> v_output_weights(HIDDEN_SIZE2 * OUTPUT_SIZE, 0.0);

     // indices for shuffling
    vector<int> indices(train_vectors.size());
    iota(indices.begin(), indices.end(), 0);

    // TRAINING

    // TESTING

    writePredictions("train_predictions.csv", train_predictions);
    writePredictions("test_predictions.csv", test_predictions);

    return 0;
}
