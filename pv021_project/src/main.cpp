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
const int EPOCHS = 5;
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

        cout << "First 5 rows of train_data: ";
for (int i = 0; i < 5; ++i) {
    for (double val : train_data.data[i]) cout << val << " ";
    cout << endl;
}

cout << "First 5 labels of train_labels: ";
for (int i = 0; i < 5; ++i) cout << train_labels.data[i][0] << " ";
cout << endl;

    //CREATE LAYERS

        Layer input_layer(784, 256);
        Layer hidden_layer2(256, 128);
        Layer hidden_layer3(128, 64);
        Layer output_layer(64, 10);

    //TRAINING 

        cout << "Train data rows: " << train_data.getRows() << ", Train labels rows: " << train_labels.getRows() << endl;

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            double total_loss = 0.0;
            int correct_predictions = 0;

            // Iterate over all training samples
            for (int i = 0; i < 50; ++i) {
                cout << "Processing sample " << i << endl;

                // Forward pass
                Matrix input = Matrix({train_data.data[i]});
                Matrix label = Matrix({train_labels.data[i]});
                
                Matrix hidden1 = input_layer.forward_relu(input);
                Matrix hidden2 = hidden_layer2.forward_relu(hidden1);
                Matrix hidden3 = hidden_layer3.forward_relu(hidden2);
                Matrix output = output_layer.forward_softmax(hidden3);

                // Loss
                Matrix loss = output.cross_entropy_loss(output, label); // Correct label passed for the loss
                total_loss += loss.data[0][0]; // Assuming loss is a single value

                // Track accuracy
                int predicted_label = distance(output.data[0].begin(), max_element(output.data[0].begin(), output.data[0].end()));
                int true_label = distance(label.data[0].begin(), max_element(label.data[0].begin(), label.data[0].end()));
                if (predicted_label == true_label) {
                    ++correct_predictions;
                }

                // Backward pass
                Matrix grad = output_layer.backward(loss, LEARNING_RATE);
                grad = hidden_layer3.backward(grad, LEARNING_RATE);
                grad = hidden_layer2.backward(grad, LEARNING_RATE);
                grad = input_layer.backward(grad, LEARNING_RATE);
            }

            // Log epoch stats
            std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS 
                    << " - Loss: " << total_loss / 50 
                    << ", Accuracy: " << 100.0 * correct_predictions / 50 << "%" << std::endl;
        }

    //TESTING
    
        Matrix predictions(test_data.getRows(), 10);
        for (int i = 0; i < test_data.getRows(); ++i) {
            Matrix input = Matrix({test_data.data[i]});
            Matrix hidden1 = input_layer.forward_relu(input);
            Matrix hidden2 = hidden_layer2.forward_relu(hidden1);
            Matrix hidden3 = hidden_layer3.forward_relu(hidden2);
            predictions.data[i] = output_layer.forward_softmax(hidden3).data[0];
        }

        double accuracy = dataset.calculate_accuracy(predictions, test_labels);
        cout << "Accuracy: " << accuracy << endl;  

    return 0;
}