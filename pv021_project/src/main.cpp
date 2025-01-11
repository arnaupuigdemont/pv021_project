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
#include "batchNorm.hh"

using namespace std;

const int OUTPUT_SIZE = 10;
const int EPOCHS = 10; 
double initial_rate = 0.001; 
double decay_rate = 0.2; 
const int BATCH_SIZE = 128; 
int lambda = 0.0;

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
        Layer input_layer(784, 512);
        BatchNorm batch_norm(512);

        Layer hidden_layer1(512, 128);
        BatchNorm batch_norm1(128);

        Layer hidden_layer2(128, 64);
        BatchNorm batch_norm2(64);

        Layer output_layer(64, 10);

    //TRAINING 
        double learning_rate = initial_rate;
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {

            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            std::cout << "Learning rate: " << learning_rate << std::endl;

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

                //Matrix input = input_layer.forward_leaky_relu(batch_inputs);
                //Matrix hidden1 = hidden_layer1.forward_leaky_relu(input);
                //Matrix hidden2 = hidden_layer2.forward_leaky_relu(hidden1);
                //Matrix hidden3 = hiddden_layer3.forward_leaky_relu(hidden2);
                //Matrix output = output_layer.forward_softmax(hidden2);

                //BATCHNORM
                // 1) Input layer: (batch_size x 784) -> (batch_size x 256)
                Matrix z_inp   = input_layer.forward(batch_inputs);       // z_inp = XW + b
                Matrix bn_inp  = batch_norm.forward(z_inp, true);         // BN sobre z_inp
                Matrix a_inp   = input_layer.leaky_relu(bn_inp);                     // activación LeakyReLU

                // 2) Hidden layer 1: (256 -> 128)
                Matrix z_h1    = hidden_layer1.forward(a_inp);            // z_h1 = a_inp * W1 + b1
                Matrix bn_h1   = batch_norm1.forward(z_h1, true);         // BN
                Matrix a_h1    = hidden_layer1.leaky_relu(bn_h1);

                // 3) Hidden layer 2: (128 -> 64)
                Matrix z_h2    = hidden_layer2.forward(a_h1);
                Matrix bn_h2   = batch_norm2.forward(z_h2, true);
                Matrix a_h2    = hidden_layer2.leaky_relu(bn_h2);

                // 4) Output layer: (64 -> 10)
                Matrix logits  = output_layer.forward(a_h2);              // z_out = a_h2 * W_out + b_out
                Matrix output   = output_layer.softmax(logits);                         // softmax para obtener probas

                // Loss
                double batch_loss = output.cross_entropy_loss(output, batch_labels);
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
                
                //BACKPROPAGATION 
                //Matrix grad = output_layer.backward_output(grad_output, learning_rate);
                //grad = hiddden_layer3.backward_relu(grad, learning_rate);
                //grad = hidden_layer2.backward_relu(grad, learning_rate);
                //grad = hidden_layer1.backward_relu(grad, learning_rate);
                //grad = input_layer.backward_relu(grad, learning_rate);

                //ADAM
                //Matrix grad = output_layer.backward_ADAM_output(grad_output, learning_rate, lambda);
                //grad = hiddden_layer3.backward_ADAM_relu(grad, learning_rate, lambda);
                //grad = hidden_layer2.backward_ADAM_relu(grad, learning_rate, lambda);
                //grad = hidden_layer1.backward_ADAM_relu(grad, learning_rate, lambda);
                //grad = input_layer.backward_ADAM(grad, learning_rate, lambda);

                //SGD with momentum
                //Matrix grad = output_layer.backward_SGD_momentum_output(grad_output, learning_rate, lambda);
                //grad = hiddden_layer3.backward_SGD_momentum_relu(grad, learning_rate, lambda);
                //grad = hidden_layer2.backward_SGD_momentum_relu(grad, learning_rate, lambda);
                //grad = hidden_layer1.backward_SGD_momentum_relu(grad, learning_rate, lambda);
                //grad = input_layer.backward_SGD_momentum_relu(grad, learning_rate, lambda);
            
                //BATCHNORM 
                Matrix grad_out = output_layer.backward_ADAM_output(grad_output, learning_rate, lambda);
                // 5) batch_norm2 backward
                Matrix grad_bn2 = batch_norm2.backward(grad_out, learning_rate);
                Matrix grad_h2 = hidden_layer2.leaky_relu_backward(grad_bn2, bn_h2, 0.01); 
                Matrix grad_a_h1 = hidden_layer2.backward_ADAM(grad_h2, learning_rate, lambda);

                // 6) batch_norm1 backward
                Matrix grad_bn1 = batch_norm1.backward(grad_a_h1, learning_rate);
                Matrix grad_h1 = hidden_layer1.leaky_relu_backward(grad_bn1, bn_h1, 0.01);
                Matrix grad_a_inp = hidden_layer1.backward_ADAM(grad_h1, learning_rate, lambda);
           
                // 9) batch_norm backward
                Matrix grad_bn_inp = batch_norm.backward(grad_a_inp, learning_rate);
                Matrix grad_z_inp = input_layer.leaky_relu_backward(grad_bn_inp, bn_inp, 0.01);
                Matrix grad_in = input_layer.backward_ADAM(grad_z_inp, learning_rate, lambda);
            
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
            
            Matrix input = input_layer.forward_leaky_relu(Matrix({test_data.data[i]}));
            Matrix hidden1 = hidden_layer1.forward_leaky_relu(input);
            Matrix hidden2 = hidden_layer2.forward_leaky_relu(hidden1);
            //Matrix hidden3 = hiddden_layer3.forward_leaky_relu(hidden2);
            predictions.data[i] = output_layer.forward_softmax(hidden2).data[0];
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_duration = total_end - total_start;
        std::cout << "Tiempo total de entrenamiento y testing: " << total_duration.count() << " segundos\n";

        // Calcular Accuracy
        double accuracy = dataset.calculate_accuracy(predictions, test_labels);
        std::cout << "Accuracy on test set: " << accuracy << "%" << endl;

    return 0;
}