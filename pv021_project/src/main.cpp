#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#include <chrono>

#include "matrix.hh"
#include "dataset.hh"
#include "layer.hh"

using namespace std;

const int OUTPUT_SIZE = 10;
const int EPOCHS = 4;
const double LEARNING_RATE = 0.001;
const int BATCH_SIZE = 64;

Matrix to_one_hot(int label, int num_classes) {
    std::vector<double> one_hot(num_classes, 0.0);
    one_hot[label] = 1.0; // Set the correct class index to 1.0
    return Matrix({one_hot});
}

int main() {

    auto total_start = std::chrono::high_resolution_clock::now();

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

    // TRAINING
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
cout << 1 << endl;
            double total_loss = 0.0;
            int correct_predictions = 0;
cout << 2 << endl;
            // Shuffle the training data at the start of each epoch
            vector<int> indices(train_data.getRows());
            iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., train_data.getRows() - 1
            random_shuffle(indices.begin(), indices.end());
cout << 3 << endl;
            std::cout << "Shuffling completed" << std::endl;

            Matrix batch_inputs(BATCH_SIZE, train_data.getCols());
            Matrix batch_labels(BATCH_SIZE, OUTPUT_SIZE);
cout << 4 << endl;
            // Iterate over batches
            for (int batch_start = 0; batch_start < train_data.getRows(); batch_start += BATCH_SIZE) {
                std::cout << "Processing batch starting at index " << batch_start << std::endl;
cout << 5 << endl;
                int batch_end = min(batch_start + BATCH_SIZE, train_data.getRows());
                int batch_size = batch_end - batch_start; // Adjust batch size for the last batch
cout << 6 << endl;
                // Create batches dynamically based on actual batch size
                Matrix batch_inputs(batch_size, train_data.getCols());
                Matrix batch_labels(batch_size, OUTPUT_SIZE);
cout << 7 << endl;
                // Prepare the batch
                for (int i = batch_start; i < batch_end; ++i) {
                    int data_index = indices[i];
                    if (data_index < 0 || data_index >= train_data.getRows()) {
                        cerr << "Index out of bounds: " << data_index << endl;
                        exit(EXIT_FAILURE);
                    }
cout << 8 << endl;
                    // Copy data to the batch
                    batch_inputs.data[i - batch_start] = train_data.data[data_index];
                    batch_labels.data[i - batch_start] = to_one_hot(train_labels.data[data_index][0], OUTPUT_SIZE).data[0];
cout << 9 << endl;
                    std::cout << "Processed sample index " << data_index << " for batch" << std::endl;
                }

                // Forward pass for the batch
                Matrix hidden1 = input_layer.forward_leaky_relu(batch_inputs);
                Matrix hidden2 = hidden_layer2.forward_leaky_relu(hidden1);
                Matrix hidden3 = hidden_layer3.forward_leaky_relu(hidden2);
                Matrix output = output_layer.forward_softmax(hidden3);

                // Compute batch loss
                double batch_loss = 0.0;
                for (int i = 0; i < batch_size; ++i) { // Use batch_size instead of BATCH_SIZE
                    batch_loss += output.cross_entropy_loss(Matrix({output.data[i]}), Matrix({batch_labels.data[i]}));
                }
                total_loss += batch_loss;

                // Track accuracy for the batch
                for (int i = 0; i < batch_size; ++i) { // Use batch_size instead of BATCH_SIZE
                    int predicted_label = distance(output.data[i].begin(), max_element(output.data[i].begin(), output.data[i].end()));
                    int true_label = distance(batch_labels.data[i].begin(), max_element(batch_labels.data[i].begin(), batch_labels.data[i].end()));
                    if (predicted_label == true_label) {
                        ++correct_predictions;
                    }
                }

                // Backward pass for the batch
                Matrix grad_output(batch_size, OUTPUT_SIZE); // Adjusted to actual batch size
                for (int i = 0; i < batch_size; ++i) {
                    for (int j = 0; j < grad_output.getCols(); ++j) {
                        grad_output.data[i][j] = output.data[i][j] - batch_labels.data[i][j];
                    }
                }
                grad_output = grad_output / batch_size; // Normalize gradients by batch size

                Matrix grad = output_layer.backward(grad_output, LEARNING_RATE);
                grad = hidden_layer3.backward(grad, LEARNING_RATE);
                grad = hidden_layer2.backward(grad, LEARNING_RATE);
                grad = input_layer.backward(grad, LEARNING_RATE);
            }

            auto epoch_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;

            // Log epoch stats
            std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS 
            << " - Loss: " << total_loss / train_data.getRows()
            << ", Accuracy: " << 100.0 * correct_predictions / train_data.getRows() 
            << "%, Duration: " << epoch_duration.count() << " seconds" << endl;
        }

    //TESTING
    
        Matrix predictions(test_data.getRows(), 10);

        for (int i = 0; i < test_data.getRows(); ++i) {
            Matrix input = Matrix({test_data.data[i]});
            Matrix hidden1 = input_layer.forward_leaky_relu(input);
            Matrix hidden2 = hidden_layer2.forward_leaky_relu(hidden1);
            Matrix hidden3 = hidden_layer3.forward_leaky_relu(hidden2);
            predictions.data[i] = output_layer.forward_softmax(hidden3).data[0];
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_duration = total_end - total_start;
        std::cout << "Tiempo total de entrenamiento y testing: " << total_duration.count() << " segundos\n";

        // Calcular Accuracy
        double accuracy = dataset.calculate_accuracy(predictions, test_labels);
        cout << "Accuracy on test set: " << accuracy << "%" << endl;

    return 0;
}