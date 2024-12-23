#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#include "matrix.hh"
#include "dataset.hh"
#include "layer.hh"

using namespace std;

const int INPUT_SIZE = 784;
const int OUTPUT_SIZE = 10;
const int EPOCHS = 10;
const double LEARNING_RATE = 0.001;
const int BATCH_SIZE = 64;

int main() {

    // LOAD DATA
        Dataset dataset;
      
        Matrix train_data = dataset.read_csv("data/fashion_mnist_train_vectors.csv");
        Matrix train_labels = dataset.read_labels("data/fashion_mnist_train_labels.csv");

        Matrix test_data = dataset.read_csv("data/fashion_mnist_test_vectors.csv");
        Matrix test_labels = dataset.read_labels("data/fashion_mnist_test_labels.csv");

    //NORMALIZE DATA MIN MAX

        train_data.normalize();
        test_data.normalize();

    //CREATE LAYERS

        Layer input_layer(784, 256);
        Layer hidden_layer2(256, 128);
        Layer hidden_layer3(128, 64);
        Layer output_layer(64, 10);

    //TRAINING 

        vector<pair<Matrix, Matrix>> batches = dataset.create_batches(train_data, train_labels, BATCH_SIZE);

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            cout << "Epoch " << epoch + 1 << endl;

            for (const auto &batch : batches) {
                const Matrix &batch_data = batch.first;
                const Matrix &batch_labels = batch.second;

                for (int i = 0; i < batch_data.getRows(); ++i) {
                    Matrix input = Matrix({batch_data.data[i]});
                    Matrix label = Matrix(1, 10);
                    label.data[0][static_cast<int>(batch_labels.data[i][0])] = 1.0;

                    // Forward pass
                    Matrix hidden1 = input_layer.forward(input);
                    Matrix hidden2 = hidden_layer2.forward(hidden1);
                    Matrix hidden3 = hidden_layer3.forward(hidden2);
                    Matrix output = output_layer.forward(hidden3);

                    // Loss
                    Matrix loss = loss.cross_entropy_loss(output, label);

                    // Backward pass
                    Matrix grad = output_layer.backward(loss, LEARNING_RATE);
                    grad = hidden_layer3.backward(grad, LEARNING_RATE);
                    grad = hidden_layer2.backward(grad, LEARNING_RATE);
                    grad = input_layer.backward(grad, LEARNING_RATE);
                }
            }
        }

    //TESTING
    
        Matrix predictions(test_data.getRows(), 10);
        for (int i = 0; i < test_data.getRows(); ++i) {
            Matrix input = Matrix({test_data.data[i]});
            Matrix hidden1 = input_layer.forward(input);
            Matrix hidden2 = hidden_layer2.forward(hidden1);
            Matrix hidden3 = hidden_layer3.forward(hidden2);
            predictions.data[i] = output_layer.forward(hidden3).data[0];
        }

        double accuracy = dataset.calculate_accuracy(predictions, test_labels);
        cout << "Accuracy: " << accuracy << endl;  

    return 0;
}