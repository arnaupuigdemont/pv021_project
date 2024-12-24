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

            double total_loss = 0.0;
            int correct_predictions = 0;

            // Shuffle the training data at the start of each epoch
            vector<int> indices(train_data.getRows());
            iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., train_data.getRows() - 1
            random_shuffle(indices.begin(), indices.end());

            std::cout << "Shuffling completed" << std::endl;

            

            // Iterate over batches
            for (int batch_start = 0; batch_start < train_data.getRows(); batch_start += BATCH_SIZE) {
                std::cout << "Processing batch starting at index " << batch_start << std::endl;

                // Calcular el final del batch y su tamaño
                int batch_end = std::min(batch_start + BATCH_SIZE, train_data.getRows());
                int batch_size = batch_end - batch_start; // Tamaño real del batch

                // Verificar el tamaño del batch
                if (batch_size <= 0) {
                    std::cerr << "Invalid batch size: " << batch_size << std::endl;
                    exit(EXIT_FAILURE);
                }

                // Crear dinámicamente las matrices para el batch
                Matrix batch_inputs(batch_size, train_data.getCols());
                Matrix batch_labels(batch_size, OUTPUT_SIZE);

                // Rellenar el batch
                for (int i = batch_start; i < batch_end; ++i) {
                    int data_index = indices[i]; // Índice después de hacer shuffle
                    if (data_index < 0 || data_index >= train_data.getRows()) {
                        std::cerr << "Index out of bounds: " << data_index << std::endl;
                        exit(EXIT_FAILURE);
                    }

                    // Copiar datos a las matrices del batch
                    batch_inputs.data[i - batch_start] = train_data.data[data_index];
                    batch_labels.data[i - batch_start] = to_one_hot(train_labels.data[data_index][0], OUTPUT_SIZE).data[0];

                    std::cout << "Processed sample index " << data_index << " for batch" << std::endl;
                }

                // Validar las dimensiones de las matrices
                if (batch_inputs.getRows() != batch_size || batch_labels.getRows() != batch_size) {
                    std::cerr << "Batch dimensions mismatch: inputs rows " << batch_inputs.getRows()
                            << ", labels rows " << batch_labels.getRows()
                            << ", expected " << batch_size << std::endl;
                    exit(EXIT_FAILURE);
                }

std::cout << "Batch inputs dimensions: " << batch_inputs.getRows() << " x " << batch_inputs.getCols() << std::endl;
std::cout << "Input layer weights dimensions: " << input_layer.getWeights().getRows() << " x " << input_layer.getWeights().getCols() << std::endl;
std::cout << "Input layer biases dimensions: " << input_layer.getBiases().getRows() << " x " << input_layer.getBiases().getCols() << std::endl;

                // **Forward pass**
                Matrix hidden1 = input_layer.forward_leaky_relu(batch_inputs);
                std::cout << "Hidden1 dimensions: " << hidden1.getRows() << " x " << hidden1.getCols() << std::endl;
                Matrix hidden2 = hidden_layer2.forward_leaky_relu(hidden1);
                std::cout << "Hidden2 dimensions: " << hidden2.getRows() << " x " << hidden2.getCols() << std::endl;
                Matrix hidden3 = hidden_layer3.forward_leaky_relu(hidden2);
                std::cout << "Hidden3 dimensions: " << hidden3.getRows() << " x " << hidden3.getCols() << std::endl;
                Matrix output = output_layer.forward_softmax(hidden3);
                std::cout << "Output dimensions: " << output.getRows() << " x " << output.getCols() << std::endl;
cout << 2 << endl;
                // **Calcular pérdida del batch**
                double batch_loss = 0.0;
                for (int i = 0; i < batch_size; ++i) {
                    batch_loss += output.cross_entropy_loss(
                        Matrix({output.data[i]}),
                        Matrix({batch_labels.data[i]})
                    );
                }
                total_loss += batch_loss;
cout << 3 << endl;
                // **Calcular precisión del batch**
                for (int i = 0; i < batch_size; ++i) {
                    int predicted_label = distance(
                        output.data[i].begin(),
                        max_element(output.data[i].begin(), output.data[i].end())
                    );
                    int true_label = distance(
                        batch_labels.data[i].begin(),
                        max_element(batch_labels.data[i].begin(), batch_labels.data[i].end())
                    );
                    if (predicted_label == true_label) {
                        ++correct_predictions;
                    }
                }
cout << 4 << endl;
                // **Backward pass**
                Matrix grad_output(batch_size, OUTPUT_SIZE);
                for (int i = 0; i < batch_size; ++i) {
                    for (int j = 0; j < grad_output.getCols(); ++j) {
                        grad_output.data[i][j] = output.data[i][j] - batch_labels.data[i][j];
                    }
                }
                grad_output = grad_output / batch_size; // Normalizar gradientes por tamaño del batch

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