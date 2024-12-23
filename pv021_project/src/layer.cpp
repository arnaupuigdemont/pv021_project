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
        // Obtener el valor máximo para estabilidad numérica
        double max_val = *std::max_element(input.data[i].begin(), input.data[i].end());
        std::cout << "Row " << i << " max_val: " << max_val << std::endl;

        double sum = 0.0;
        for (int j = 0; j < input.getCols(); ++j) {
            // Aplicar recorte para evitar problemas numéricos
            double raw_val = input.data[i][j] - max_val;
            double clipped_val = std::max(-20.0, std::min(raw_val, 20.0));
            double exp_val = exp(clipped_val);
            sum += exp_val;

            result.data[i][j] = exp_val;

            // Comprobación de los valores intermedios
            std::cout << "Row " << i << ", Col " << j << " raw_val: " << raw_val
                      << ", clipped_val: " << clipped_val
                      << ", exp_val: " << exp_val << std::endl;
        }

        // Comprobación de la suma de exponenciales
        if (sum <= 0 || std::isnan(sum) || std::isinf(sum)) {
            std::cerr << "Error: Sum of exponentials is invalid (sum = " << sum << ") in Row " << i << std::endl;
        } else {
            std::cout << "Row " << i << " sum of exp: " << sum << std::endl;
        }

        for (int j = 0; j < input.getCols(); ++j) {
            result.data[i][j] /= sum;

            // Comprobación del resultado final de softmax
            if (std::isnan(result.data[i][j]) || std::isinf(result.data[i][j])) {
                std::cerr << "Error: Invalid softmax value at Row " << i << ", Col " << j
                          << " (value = " << result.data[i][j] << ")" << std::endl;
            }
        }

        // Comprobación de que la suma de las probabilidades es aproximadamente 1
        double row_sum = std::accumulate(result.data[i].begin(), result.data[i].end(), 0.0);
        if (std::abs(row_sum - 1.0) > 1e-6) {
            std::cerr << "Warning: Row " << i << " probabilities do not sum to 1 (sum = " << row_sum << ")" << std::endl;
        } else {
            std::cout << "Row " << i << " probabilities sum to: " << row_sum << std::endl;
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
