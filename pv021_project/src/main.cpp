#include "dataset.hpp"
#include "network.hpp"
#include <iostream>

int main() {

    std::cout << "Starting..." << std::endl;
    dataset reader;
    std::cout << "Reading data..." << std::endl;
    auto trainValues = reader.readCSVValues("data/fashion_mnist_train_vectors.csv");
    std::cout << "Reading labels..." << std::endl;
    auto trainLabels = reader.readCSVLabels("data/fashion_mnist_train_labels.csv");
    

    MLP network(784);
    network.addLayer(128, activations::_leakyReLu);
    network.addLayer(64, activations::_leakyReLu);
    network.addLayer(10, activations::_softmax);

    std::cout << "Training..." << std::endl;
    network.train(trainValues, trainLabels, 0.001, 15, 512);

    std::cout << "Predicting..." << std::endl;
    auto testValues = reader.readCSVValues("data/fashion_mnist_test_vectors.csv");
    auto predictedTestLabels = network.predict(testValues);   
	reader.exportResults("/actualPredictions.csv", predictedTestLabels);

    return 0;
}

