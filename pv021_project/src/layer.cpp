#include "layer.hh"
      
    //PUBLIC

        Layer::Layer(int input_size, int output_size) 
            : weights(Matrix::HeIni(input_size, output_size, input_size)), 
            biases(Matrix(1, output_size, 0.01)),                              
            cached_input(Matrix(0, 0)) {}


        Matrix Layer::forward_relu(const Matrix &input) {
            cached_input = input;
            return relu((input * weights) + biases);
        }

        Matrix Layer::forward_leaky_relu(const Matrix &input) {
            cached_input = input;
            return leaky_relu((input * weights) + biases);
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

        Matrix Layer::getWeights() {
            return weights;
        }

        Matrix Layer::getBiases() {
            return biases;
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
                // Obtener el valor máximo para estabilidad numérica
                double max_val = *max_element(input.data[i].begin(), input.data[i].end());

                double sum = 0.0;
                for (int j = 0; j < input.getCols(); ++j) {
                    // Aplicar recorte para evitar problemas numéricos
                    double clipped_val = max(-20.0, std::min(input.data[i][j] - max_val, 20.0));
                    double exp_val = exp(clipped_val);
                    sum += exp_val;
                    result.data[i][j] = exp_val;
                }

                for (int j = 0; j < input.getCols(); ++j) {
                    result.data[i][j] /= sum;
                }
            }
            return result;
        }

        Matrix Layer::leaky_relu(const Matrix &input) {
            Matrix result(input.getRows(), input.getCols());
            for (int i = 0; i < input.getRows(); ++i)
                for (int j = 0; j < input.getCols(); ++j)
                    result.data[i][j] = (input.data[i][j] > 0) ? input.data[i][j] : 0.01 * input.data[i][j];
            return result;
        }
