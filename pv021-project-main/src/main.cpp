#include "dataset.hpp"
#include "network.hpp"
#include "tests.hpp"

int main() {

    CSVReader reader;
    auto trainValues = reader.readCSVValues("../data/fashion_mnist_train_vectors.csv");
    auto trainLabels = reader.readCSVLabels("../data/fashion_mnist_train_labels.csv");
    

    MLP network(784);
    network.addLayer(128, activations::_leakyReLu);
    network.addLayer(64, activations::_leakyReLu);
    network.addLayer(10, activations::_softmax);


    network.train(trainValues, trainLabels, 0.001, 15, 512);


    auto testValues = reader.readCSVValues("../data/fashion_mnist_test_vectors.csv");
    auto predictedTestLabels = network.predict(testValues);   
	reader.exportResults("../actualPredictions", predictedTestLabels);

    return 0;
}

