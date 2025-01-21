#include "dataset.hpp"
#include "network.hpp"
#include <iostream>

int main() {

 
    // read and normalize data
    std::cout << "Starting..." << std::endl;
    dataset ds;
    std::cout << "Reading data..." << std::endl;
    auto trainValues = ds.readValues("data/fashion_mnist_train_vectors.csv");
    auto testValues = ds.readValues("data/fashion_mnist_test_vectors.csv");
    std::cout << "Reading labels..." << std::endl;
    auto trainLabels = ds.readLabels("data/fashion_mnist_train_labels.csv");  
    auto testLabels = ds.readLabels("data/fashion_mnist_test_labels.csv");
    
    // create and train network
    Network nn(784);
    nn.addLayer(256);
    nn.addLayer(128);
    nn.addOutputLayer(10);

    std::cout << "Training..." << std::endl;
    nn.train(trainValues, trainLabels, 0.001, 4, 128);

    // predict and calculate accuracy
    std::cout << "Predicting..." << std::endl;
    auto predictedTestLabels = nn.predict(testValues);   
    auto predictedTrainLabels = nn.predict(trainValues);
	ds.writeResults("test_predictions.csv", predictedTestLabels);
    ds.writeResults("train_predictions.csv", predictedTrainLabels);

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

