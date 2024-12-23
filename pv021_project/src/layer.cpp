#include "layer.hh"
      
    //PUBLIC

        Matrix weights;
        Matrix biases;

        Layer::Layer(int input_size, int output_size) 
            : weights(Matrix::Random(input_size, output_size)), 
              biases(Matrix::Random(1, output_size)), 
              cached_input(Matrix(0, 0)) {}

        Matrix Layer::forward(const Matrix &input) {
            cached_input = input;
            return relu((input * weights) + biases);
        }

        Matrix Layer::backward(const Matrix &grad_output, double learning_rate) {
            Matrix grad_input = grad_output * weights.transpose();
            Matrix grad_weights = cached_input.transpose() * grad_output;

            // Update weights and biases
            weights = weights - grad_weights.scalar_mul(learning_rate);
            for (int i = 0; i < biases.getRows(); ++i)
                for (int j = 0; j < biases.getCols(); ++j)
                    biases.data[i][j] -= learning_rate * grad_output.data[i][j];

            return grad_input;
        }

    //PRIVATE
        Matrix cached_input;

        Matrix Layer::relu(const Matrix &input) {
            Matrix result(input.getRows(), input.getCols());
            for (int i = 0; i < input.getRows(); ++i)
                for (int j = 0; j < input.getCols(); ++j)
                    result.data[i][j] = max(0.0, input.data[i][j]);
            return result;
        }
