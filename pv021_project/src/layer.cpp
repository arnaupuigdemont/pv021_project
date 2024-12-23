#include "layer.hh"
      
    //PUBLIC

        Layer::Layer(int input_size, int output_size) 
            : weights(Matrix::Xavier(input_size, output_size, input_size)), 
            biases(Matrix(1, output_size)),                              
            cached_input(Matrix(0, 0)) {}


        Matrix Layer::forward_relu(const Matrix &input) {
            cached_input = input;
            return relu((input * weights) + biases);
        }

        Matrix Layer::forward_softmax(const Matrix &input) {
            cached_input = input;
            return softmax((input * weights) + biases);
        }

        Matrix Layer::backward(const Matrix &grad_output, double learning_rate) {
            Matrix grad_input = grad_output * weights.transpose();
            Matrix grad_weights = cached_input.transpose() * grad_output;

            Matrix grad_biases(1, grad_output.getCols());
            for (int j = 0; j < grad_output.getCols(); ++j) {
                for (int i = 0; i < grad_output.getRows(); ++i) {
                    grad_biases.data[0][j] += grad_output.data[i][j];
                }
            }

            weights = weights - grad_weights.scalar_mul(learning_rate);
            biases = biases - grad_biases.scalar_mul(learning_rate);

            return grad_input;
        }

    //PRIVATE

        Matrix Layer::relu(const Matrix &input) {
            Matrix result(input.getRows(), input.getCols());
            for (int i = 0; i < input.getRows(); ++i)
                for (int j = 0; j < input.getCols(); ++j)
                    result.data[i][j] = max(0.0, input.data[i][j]);
            return result;
        }

        Matrix Layer::softmax(const Matrix &input) {
            Matrix result(input.getRows(), input.getCols());
            for (int i = 0; i < input.getRows(); ++i) {
                double sum = 0.0;
                for (int j = 0; j < input.getCols(); ++j)
                    sum += exp(input.data[i][j]);
                for (int j = 0; j < input.getCols(); ++j)
                    result.data[i][j] = exp(input.data[i][j]) / sum;
            }
            return result;
        }
