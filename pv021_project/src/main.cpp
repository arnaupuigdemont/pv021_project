#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <map>

#include "aux_func.h"
#include "normalization_func.h"
#include "neural_net_func.h"
#include "activation_func.h"

using namespace std;

const int INPUT_SIZE = 784;
const int HIDDEN_SIZE_LAYER_1 = 64;
const int HIDDEN_SIZE_LAYER_2 = 64;
const int OUTPUT_SIZE = 12;
const int EPOCHS = 12;
const int BATCH_SIZE = 32;

int main() {
    // Load training data
    vector<vector<double>> train_vectors = read_csv("data/fashion_mnist_train_vectors.csv");
    vector<int> train_labels = load_labels("data/fashion_mnist_train_labels.csv");

    // Load test data
    vector<vector<double>> test_vectors = read_csv("data/fashion_mnist_test_vectors.csv");
    vector<int> test_labels = load_labels("data/fashion_mnist_test_labels.csv");

    //min max normalization
    pair<vector<double>, vector<double>> min_max = find_min_max(train_vectors);

    normalize(train_vectors, min_max.first, min_max.second);
    normalize(test_vectors, min_max.first, min_max.second);

    //validation split 
    int total_training = train_vectors.size();
    int validation_size = total_training / 10;
    int training_size = total_training - validation_size;

    //hiddden and output vectors
    vector<double> hidden_vector_layer1(HIDDEN_SIZE_LAYER_1);
    vector<double> hidden_vector_layer2(HIDDEN_SIZE_LAYER_2);
    vector<double> output_vector(OUTPUT_SIZE);

    // Weights and biases
    random_device rd;
    mt19937 gen(3);
    uniform_real_distribution<> dis(-0.1, 0.1);

    // Weights for the first hidden layer
    vector<double> hidden_weights_layer1(INPUT_SIZE * HIDDEN_SIZE_LAYER_1);
    for (auto& weight : hidden_weights_layer1) {
        weight = dis(gen);
    }

    // Weights for the second hidden layer
    vector<double> hidden_weights_layer2(HIDDEN_SIZE_LAYER_1 * HIDDEN_SIZE_LAYER_2);
    for (auto& weight : hidden_weights_layer2) {
        weight = dis(gen);
    }

    // Weights for the output layer
    vector<double> output_weights(HIDDEN_SIZE_LAYER_2 * OUTPUT_SIZE);
    for (auto& weight : output_weights) {
        weight = dis(gen);
    }

    // Biases for the first hidden layer
    vector<double> hidden_bias_layer1(HIDDEN_SIZE_LAYER_1);
    for (auto& bias : hidden_bias_layer1) {
        bias = dis(gen);
    }

    // Biases for the second hidden layer
    vector<double> hidden_bias_layer2(HIDDEN_SIZE_LAYER_2);
    for (auto& bias : hidden_bias_layer2) {
        bias = dis(gen);
    }

    // Biases for the output layer
    vector<double> output_bias(OUTPUT_SIZE);
    for (auto& bias : output_bias) {
        bias = dis(gen);
    }

    // adam weights
    vector<double> m_hidden_weights1(INPUT_SIZE * HIDDEN_SIZE_LAYER_1, 0.0);
    vector<double> v_hidden_weights1(INPUT_SIZE * HIDDEN_SIZE_LAYER_1, 0.0);
    vector<double> m_hidden_weights2(HIDDEN_SIZE_LAYER_1 * HIDDEN_SIZE_LAYER_2, 0.0);
    vector<double> v_hidden_weights2(HIDDEN_SIZE_LAYER_1 * HIDDEN_SIZE_LAYER_2, 0.0);
    vector<double> m_output_weights(HIDDEN_SIZE_LAYER_2 * OUTPUT_SIZE, 0.0);
    vector<double> v_output_weights(HIDDEN_SIZE_LAYER_2 * OUTPUT_SIZE, 0.0);

    map<int, int> train_predictions_map;

     // indices for shuffling
    vector<int> indices(train_vectors.size());
    for (vector<vector<double>>::size_type i = 0; i < train_vectors.size(); ++i) {
        indices[i] = i;
    }

    // TRAINING
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        shuffle(indices.begin(), indices.end(), gen);
        
        for (int idx = 0; idx < training_size; idx += BATCH_SIZE) {
    
            int current_batch_size = min(BATCH_SIZE, training_size - idx);

            // batch vectors
            vector<vector<double>> batch_input(current_batch_size, vector<double>(INPUT_SIZE, 0.0));
            vector<vector<double>> batch_hidden1(current_batch_size, vector<double>(HIDDEN_SIZE_LAYER_1, 0.0));
            vector<vector<double>> batch_hidden2(current_batch_size, vector<double>(HIDDEN_SIZE_LAYER_2, 0.0));
            vector<vector<double>> batch_output(current_batch_size, vector<double>(OUTPUT_SIZE, 0.0));
            vector<vector<double>> batch_d_hidden1(current_batch_size, vector<double>(HIDDEN_SIZE_LAYER_1, 0.0));
            vector<vector<double>> batch_d_hidden2(current_batch_size, vector<double>(HIDDEN_SIZE_LAYER_2, 0.0));
            vector<vector<double>> batch_error_output(current_batch_size, vector<double>(OUTPUT_SIZE, 0.0));

            // input batch
            for (long unsigned int b = 0; b < batch_input.size(); ++b) {
                batch_input[b] = train_vectors[indices[idx + b]];
            }

            // forward pass
            pass_hidden(batch_input, batch_hidden1, hidden_weights_layer1, hidden_bias_layer1, relu);
            pass_hidden(batch_hidden1, batch_hidden2, hidden_weights_layer2, hidden_bias_layer2, relu);
            pass_output(batch_hidden2, batch_output, output_weights, output_bias, softmax);

            // compute error and b
            for (long unsigned int b = 0; b < batch_input.size(); ++b) {
                int i = indices[idx + b];
                int true_label = train_labels[i];
                for (int o = 0; o < OUTPUT_SIZE; ++o) {
                    batch_error_output[b][o] = (o == true_label) ? 1.0 - batch_output[b][o] : -batch_output[b][o];
                }
            }

            backpropagation_hidden(batch_hidden2, batch_d_hidden2, batch_error_output, output_weights, leaky_relu);
            backpropagation_hidden(batch_hidden1, batch_d_hidden1, batch_d_hidden2, hidden_weights_layer2, leaky_relu);

            // update weights
            update_weights_Adam(hidden_weights_layer1, hidden_bias_layer1, batch_d_hidden1, batch_input,  m_hidden_weights1, v_hidden_weights1, epoch);
            update_weights_Adam(hidden_weights_layer2, hidden_bias_layer2, batch_d_hidden2, batch_hidden1,  m_hidden_weights2, v_hidden_weights2, epoch);
            update_weights_Adam(output_weights, output_bias, batch_error_output, batch_hidden2, m_output_weights, v_output_weights, epoch);

            // push label to predictions
            if (epoch == EPOCHS) {
                for (long unsigned int b = 0; b < batch_input.size(); ++b) {
                    int i = indices[idx + b];
                    int predicted_label = max_element(batch_output[b].begin(), batch_output[b].end()) - batch_output[b].begin();
                    train_predictions_map[i] = predicted_label;
                }
            }
        }

        // VALIDATION
        int correct_count = 0;
        for (int i = training_size; i < total_training; i += BATCH_SIZE) {
            // size of the current batch 
            int current_batch_size = min(BATCH_SIZE, total_training - i);

            // batch vectors
            vector<vector<double>> batch_input(current_batch_size, vector<double>(INPUT_SIZE, 0.0));
            vector<vector<double>> batch_hidden1(current_batch_size, vector<double>(HIDDEN_SIZE_LAYER_1, 0.0));
            vector<vector<double>> batch_hidden2(current_batch_size, vector<double>(HIDDEN_SIZE_LAYER_2, 0.0));
            vector<vector<double>> batch_output(current_batch_size, vector<double>(OUTPUT_SIZE, 0.0));

            // input batch
            for (int b = 0; b < current_batch_size; ++b) {
                batch_input[b] = train_vectors[i + b];
            }

            // forward pass
            pass_hidden(batch_input, batch_hidden1, hidden_weights_layer1, hidden_bias_layer1, relu);
            pass_hidden(batch_hidden1, batch_hidden2, hidden_weights_layer2, hidden_bias_layer2, relu);
            pass_output(batch_hidden2, batch_output, output_weights, output_bias, softmax);

            // push label to predictions
            for (int b = 0; b < current_batch_size; ++b) {
                int predicted_label = max_element(batch_output[b].begin(), batch_output[b].end()) - batch_output[b].begin();
                if (epoch == EPOCHS) train_predictions_map[i + b] = predicted_label;
                correct_count += (predicted_label == train_labels[i + b]) ? 1 : 0;
            }
        }
        cout << "Epoch: " << epoch << ", accuracy on validation data: " << (double)correct_count / validation_size << endl;
    }

    // TESTING
    int test_correct_count = 0;
    vector<int> test_predictions;
    for (long unsigned int i = 0; i < test_vectors.size(); i += BATCH_SIZE) {
        // size of the current batch 
        int current_batch_size = min(BATCH_SIZE, static_cast<int>(test_vectors.size() - i));

        // batch vectors
        vector<vector<double>> batch_input(current_batch_size, vector<double>(INPUT_SIZE, 0.0));
        vector<vector<double>> batch_hidden1(current_batch_size, vector<double>(HIDDEN_SIZE_LAYER_1, 0.0));
        vector<vector<double>> batch_hidden2(current_batch_size, vector<double>(HIDDEN_SIZE_LAYER_2, 0.0));
        vector<vector<double>> batch_output(current_batch_size, vector<double>(OUTPUT_SIZE, 0.0));

        //input batch
        for (int b = 0; b < current_batch_size; ++b) {
            batch_input[b] = test_vectors[i + b];
        }

        // forward pass
        pass_hidden(batch_input, batch_hidden1, hidden_weights_layer1, hidden_bias_layer1, relu);
        pass_hidden(batch_hidden1, batch_hidden2, hidden_weights_layer2, hidden_bias_layer2, relu);
        pass_output(batch_hidden2, batch_output, output_weights, output_bias, softmax);

        // push label to predictions
        for (int b = 0; b < current_batch_size; ++b) {
            int predicted_label = max_element(batch_output[b].begin(), batch_output[b].end()) - batch_output[b].begin();
            test_predictions.push_back(predicted_label);
            test_correct_count += (predicted_label == test_labels[i + b]) ? 1 : 0;
        }
    }
    cout << "Accuracy on test data: " << (double)test_correct_count / test_vectors.size() << endl;

    // reorder train predictions back
    vector<int> train_predictions;
    for (int i = 0; i < total_training; ++i) {
        train_predictions.push_back(train_predictions_map[i]);
    }

    write_predictions("train_predictions.csv", train_predictions);
    write_predictions("test_predictions.csv", test_predictions);

    return 0;
}
