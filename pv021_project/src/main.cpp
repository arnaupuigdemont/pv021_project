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
    network.train(trainValues, trainLabels, 0.001, 15, 256);

    std::cout << "Predicting..." << std::endl;
    auto testValues = reader.readCSVValues("data/fashion_mnist_test_vectors.csv");
    auto predictedTestLabels = network.predict(testValues);   
    auto predictedTrainLabels = network.predict(trainValues);
	reader.exportResults("test_predictions.csv", predictedTestLabels);
    reader.exportResults("train_predictions.csv", predictedTrainLabels);

    std::cout << "Calculating accuracy..." << std::endl;
    auto testLabels = reader.readCSVLabels("data/fashion_mnist_test_labels.csv");

    int correctPredictionsTest = 0;
    for (size_t i = 0; i < testLabels.size(); ++i) {
        if (testLabels[i] == predictedTestLabels[i]) {
            ++correctPredictionsTest;
        }
    }

    int correctPredictionsTrain = 0;
    for (size_t i = 0; i < trainLabels.size(); ++i) {
        if (trainLabels[i] == predictedTrainLabels[i]) {
            ++correctPredictionsTrain;
        }
    }

    double accuracy = static_cast<double>(correctPredictionsTest) / testLabels.size();
    std::cout << "Test accuracy: " << accuracy * 100 << "%" << std::endl;
    double accuracyTrain = static_cast<double>(correctPredictionsTrain) / trainLabels.size();
    std::cout << "Train accuracy: " << accuracyTrain * 100 << "%" << std::endl;

    return 0;
}

