#include "dataset.hpp"
#include "network.hpp"
#include <iostream>

int main() {

 
    // read and normalize data
    std::cout << "Starting..." << std::endl;
    dataset reader;
    std::cout << "Reading data..." << std::endl;
    auto trainValues = reader.readValues("data/fashion_mnist_train_vectors.csv");
    auto testValues = reader.readValues("data/fashion_mnist_test_vectors.csv");
    std::cout << "Reading labels..." << std::endl;
    auto trainLabels = reader.readLabels("data/fashion_mnist_train_labels.csv");  
    auto testLabels = reader.readLabels("data/fashion_mnist_test_labels.csv");
    
    // create and train network
    MLP network(784);
    network.addLayer(128, activations::_leakyReLu);
    network.addLayer(64, activations::_leakyReLu);
    network.addLayer(10, activations::_softmax);

    std::cout << "Training..." << std::endl;
    network.train(trainValues, trainLabels, 0.001, 8, 64);

    // predict and calculate accuracy
    std::cout << "Predicting..." << std::endl;
    auto predictedTestLabels = network.predict(testValues);   
    auto predictedTrainLabels = network.predict(trainValues);
	reader.writeResults("test_predictions.csv", predictedTestLabels);
    reader.writeResults("train_predictions.csv", predictedTrainLabels);

    std::cout << "Calculating accuracy..." << std::endl;

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

