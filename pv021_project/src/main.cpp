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
const int EPOCHS = 8;
double initial_lr = 0.001; 
double decay_rate = 0.1;
const int BATCH_SIZE = 64;
int lambda = 0.0001;

Matrix to_one_hot(int label, int num_classes) {
    std::vector<double> one_hot(num_classes, 0.0);
    one_hot[label] = 1.0; // Set the correct class index to 1.0
    return Matrix({one_hot});
}

double adjust_learning_rate(double initial_lr, double decay_rate, int epoch) {
    return initial_lr / (1 + decay_rate * epoch);
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

        Layer input_layer(784, 512);
        Layer hidden_layer2(512, 256);
        Layer hidden_layer3(256, 64);
        Layer output_layer(64, 10);

    //TRAINING 

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {

            auto epoch_start = std::chrono::high_resolution_clock::now();

            double learning_rate = adjust_learning_rate(initial_lr, decay_rate, epoch);

            double total_loss = 0.0;
            int correct_predictions = 0;

            vector<int> indices(train_data.getRows());
            iota(indices.begin(), indices.end(), 0); 
            random_shuffle(indices.begin(), indices.end());

            // Iterate over all training samples
            for (int batch_start = 0; batch_start < train_data.getRows(); batch_start += BATCH_SIZE) {

                int batch_end = min(batch_start + BATCH_SIZE, train_data.getRows());
                int batch_size = batch_end - batch_start; 

                Matrix batch_inputs(batch_size, train_data.getCols());
                Matrix batch_labels(batch_size, OUTPUT_SIZE);

                for (int i = 0; i < batch_size; ++i) {
                    int data_index = indices[batch_start + i];
                    batch_inputs.data[i] = train_data.data[data_index];
                    batch_labels.data[i] = to_one_hot(train_labels.data[data_index][0], OUTPUT_SIZE).data[0];
                }

                Matrix hidden1 = input_layer.forward_leaky_relu(batch_inputs);
                //hidden1 = hidden1.apply_dropout(0.8);
                Matrix hidden2 = hidden_layer2.forward_leaky_relu(hidden1);
               // hidden2 = hidden2.apply_dropout(0.8);
                Matrix hidden3 = hidden_layer3.forward_leaky_relu(hidden2);
               // hidden3 = hidden3.apply_dropout(0.8);
                Matrix output = output_layer.forward_softmax(hidden3);

                // Loss
                double batch_loss = 0.0;
                for (int i = 0; i < batch_size; ++i) {
                    batch_loss += output.cross_entropy_loss(
                        Matrix({output.data[i]}), Matrix({batch_labels.data[i]})
                    );
                }
                double l2_penalty = lambda * (
                    input_layer.compute_l2_penalty() +
                    hidden_layer2.compute_l2_penalty() +
                    hidden_layer3.compute_l2_penalty() +
                    output_layer.compute_l2_penalty()
                );

                batch_loss += l2_penalty;
                total_loss += batch_loss;

                // Track accuracy
                for (int i = 0; i < batch_size; ++i) {
                    int predicted_label = distance(
                        output.data[i].begin(), max_element(output.data[i].begin(), output.data[i].end())
                    );
                    int true_label = distance(
                        batch_labels.data[i].begin(), max_element(batch_labels.data[i].begin(), batch_labels.data[i].end())
                    );
                    if (predicted_label == true_label) {
                        ++correct_predictions;
                    }
                }

                // Backward pass
                Matrix grad_output = output;
                for (int i = 0; i < batch_size; ++i) {
                    for (int j = 0; j < grad_output.getCols(); ++j) {
                        grad_output.data[i][j] -= batch_labels.data[i][j];
                    }
                }
                grad_output = grad_output / batch_size; // Normalizar gradientes por tamaño del batch

                Matrix grad = output_layer.backward_ADAM(grad_output, learning_rate, lambda);
                grad = grad.clip_gradients(-10.0, 10.0);
                grad = hidden_layer3.backward_ADAM(grad, learning_rate, lambda);
                grad = grad.clip_gradients(-10.0, 10.0);
                grad = hidden_layer2.backward_ADAM(grad, learning_rate, lambda);
                grad = grad.clip_gradients(-10.0, 10.0);
                grad = input_layer.backward_ADAM(grad, learning_rate, lambda);
                grad = grad.clip_gradients(-10.0, 10.0);
            }

            auto epoch_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> epoch_duration = epoch_end - epoch_start;

            // Log epoch stats
            std::cout << "Epoch " << epoch + 1 << "/" << EPOCHS 
                    << " - Loss: " << total_loss / train_data.getRows()
                    << ", Accuracy: " << 100.0 * correct_predictions / train_data.getRows() << "%" << 
                    " duration: " << epoch_duration.count() << " seconds" << endl;
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
        std::cout << "Accuracy on test set: " << accuracy << "%" << endl;

    return 0;
}