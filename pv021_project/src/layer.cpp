#include "layer.hh"
      
    //PUBLIC

        Layer::Layer(int input_size, int output_size) 
            : weights(Matrix::HeIni(input_size, output_size, input_size)), 
                biases(Matrix(1, output_size, 0.01)), 
                m_weights(input_size, output_size),    
                v_weights(input_size, output_size),
                m_biases(1, output_size),
                v_biases(1, output_size),
                momentum_weights(input_size, output_size),
                momentum_biases(1, output_size),                             
                cached_input(Matrix(0, 0)) {}


        Matrix Layer::forward_relu(const Matrix &input) {
            cached_input = input;
            return relu((input * weights) + biases);
        }

        Matrix Layer::forward_leaky_relu(const Matrix &input) {
            cached_input = input;
            Matrix res = input * weights;
            res = res.broadcast_biases(res, biases);
            return leaky_relu(res);
        }

        Matrix Layer::forward_softmax(const Matrix &input) {
            cached_input = input;
            Matrix res = input * weights;
            res = res.broadcast_biases(res, biases);
            return softmax(res);
        }

        double Layer::compute_l2_penalty() const {
            double penalty = 0.0;
            for (const auto &row : weights.data) {
                for (double weight : row) {
                    penalty += weight * weight;
                }
            }
            return penalty;
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

        Matrix Layer::backward_ADAM(const Matrix &grad_output, double learning_rate, double lambda) {

            t++; // Incrementar el paso de tiempo

            // Calcular gradientes estándar usando Leaky ReLU derivada
            Matrix grad_weights = cached_input.transpose() * grad_output;

            // Gradientes estándar para pesos y sesgos
            Matrix grad_biases(1, grad_output.getCols());
            for (int j = 0; j < grad_output.getCols(); ++j) {
                for (int i = 0; i < grad_output.getRows(); ++i) {
                    grad_biases.data[0][j] += grad_output.data[i][j];
                }
            }

            // Regularización L2
            grad_weights = grad_weights + weights.scalar_mul(lambda);

            // Actualización de momentos de Adam
            m_weights = m_weights.scalar_mul(beta1) + grad_weights.scalar_mul(1 - beta1);
            v_weights = v_weights.scalar_mul(beta2) + grad_weights.hadamard(grad_weights).scalar_mul(1 - beta2);

            m_biases = m_biases.scalar_mul(beta1) + grad_biases.scalar_mul(1 - beta1);
            v_biases = v_biases.scalar_mul(beta2) + grad_biases.hadamard(grad_biases).scalar_mul(1 - beta2);

            // Corregir sesgo
            Matrix m_weights_hat = m_weights.scalar_mul(1.0 / (1.0 - pow(beta1, t)));
            Matrix v_weights_hat = v_weights.scalar_mul(1.0 / (1.0 - pow(beta2, t)));

            Matrix m_biases_hat = m_biases.scalar_mul(1.0 / (1.0 - pow(beta1, t)));
            Matrix v_biases_hat = v_biases.scalar_mul(1.0 / (1.0 - pow(beta2, t)));

            // Actualización de los pesos y los sesgos
            weights = weights - (m_weights_hat / (v_weights_hat.sqrt() + epsilon)).scalar_mul(learning_rate);
            biases = biases - (m_biases_hat / (v_biases_hat.sqrt() + epsilon)).scalar_mul(learning_rate);

            // Gradiente de entrada
            Matrix grad_input = grad_output * weights.transpose();

            return grad_input;
        }

        Matrix Layer::backward_ADAM_hidden(const Matrix &grad_output, double learning_rate, double lambda) {
            std::cout << "Entering backward_ADAM_hidden" << std::endl;
            std::cout << "grad_output rows: " << grad_output.getRows() << ", cols: " << grad_output.getCols() << std::endl;

            // Derivada de Leaky ReLU
            Matrix grad_activation = leaky_relu_derivative(cached_input).hadamard(grad_output);
            std::cout << "Computed grad_activation" << std::endl;

            // Gradientes estándar para pesos y sesgos
            Matrix grad_weights = cached_input.transpose() * grad_activation;
            std::cout << "Computed grad_weights" << std::endl;

            Matrix grad_biases(1, grad_activation.getCols());
            for (int j = 0; j < grad_activation.getCols(); ++j) {
                for (int i = 0; i < grad_activation.getRows(); ++i) {
                    grad_biases.data[0][j] += grad_activation.data[i][j];
                }
            }
            std::cout << "Computed grad_biases" << std::endl;

            // Actualización de Adam para pesos y sesgos
            m_weights = m_weights.scalar_mul(beta1) + grad_weights.scalar_mul(1 - beta1);
            v_weights = v_weights.scalar_mul(beta2) + grad_weights.hadamard(grad_weights).scalar_mul(1 - beta2);
            m_biases = m_biases.scalar_mul(beta1) + grad_biases.scalar_mul(1 - beta1);
            v_biases = v_biases.scalar_mul(beta2) + grad_biases.hadamard(grad_biases).scalar_mul(1 - beta2);
            std::cout << "Updated m_weights and v_weights, m_biases and v_biases" << std::endl;

            // Corrección del sesgo
            Matrix m_weights_hat = m_weights.scalar_mul(1.0 / (1.0 - pow(beta1, t)));
            Matrix v_weights_hat = v_weights.scalar_mul(1.0 / (1.0 - pow(beta2, t)));
            Matrix m_biases_hat = m_biases.scalar_mul(1.0 / (1.0 - pow(beta1, t)));
            Matrix v_biases_hat = v_biases.scalar_mul(1.0 / (1.0 - pow(beta2, t)));
            std::cout << "Applied bias correction" << std::endl;

            // Actualización de parámetros
            weights = weights - (m_weights_hat / (v_weights_hat.sqrt() + epsilon)).scalar_mul(learning_rate);
            biases = biases - (m_biases_hat / (v_biases_hat.sqrt() + epsilon)).scalar_mul(learning_rate);
            std::cout << "Updated weights and biases" << std::endl;

            // Gradiente para la siguiente capa
            Matrix grad_input = grad_activation * weights.transpose(); // Tamaño resultante: (128, 256)
            std::cout << "Computed grad_input: (" << grad_input.getRows() << ", " << grad_input.getCols() << ")" << std::endl;

            return grad_input;
        }

        Matrix Layer::backward_ADAM_output(const Matrix &output, const Matrix &target, double learning_rate, double lambda) {
            // Gradiente de la pérdida (Cross-Entropy con Softmax)
            Matrix grad_output = output - target;

            // Gradientes estándar para pesos y sesgos
            Matrix grad_weights = cached_input.transpose() * grad_output;
            Matrix grad_biases(1, grad_output.getCols());
            for (int j = 0; j < grad_output.getCols(); ++j) {
                for (int i = 0; i < grad_output.getRows(); ++i) {
                    grad_biases.data[0][j] += grad_output.data[i][j];
                }
            }

            // Actualización de Adam para pesos y sesgos
            m_weights = m_weights.scalar_mul(beta1) + grad_weights.scalar_mul(1 - beta1);
            v_weights = v_weights.scalar_mul(beta2) + grad_weights.hadamard(grad_weights).scalar_mul(1 - beta2);
            m_biases = m_biases.scalar_mul(beta1) + grad_biases.scalar_mul(1 - beta1);
            v_biases = v_biases.scalar_mul(beta2) + grad_biases.hadamard(grad_biases).scalar_mul(1 - beta2);

            // Corregir sesgo
            Matrix m_weights_hat = m_weights.scalar_mul(1.0 / (1.0 - pow(beta1, t)));
            Matrix v_weights_hat = v_weights.scalar_mul(1.0 / (1.0 - pow(beta2, t)));
            Matrix m_biases_hat = m_biases.scalar_mul(1.0 / (1.0 - pow(beta1, t)));
            Matrix v_biases_hat = v_biases.scalar_mul(1.0 / (1.0 - pow(beta2, t)));

            // Actualización de parámetros
            weights = weights - (m_weights_hat / (v_weights_hat.sqrt() + epsilon)).scalar_mul(learning_rate);
            biases = biases - (m_biases_hat / (v_biases_hat.sqrt() + epsilon)).scalar_mul(learning_rate);

            // Transformar el gradiente para ajustarse a la capa anterior
            Matrix grad_input = grad_output * weights.transpose(); // Tamaño resultante: (128, 64)
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

        Matrix Layer::softmax_derivative(const Matrix &input) {
            Matrix result(input.getRows(), input.getCols());
            for (int i = 0; i < input.getRows(); ++i) {
                for (int j = 0; j < input.getCols(); ++j) {
                    result.data[i][j] = input.data[i][j] * (1 - input.data[i][j]);
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

        Matrix Layer::leaky_relu_derivative(const Matrix &input) {
            Matrix result(input.getRows(), input.getCols());
            for (int i = 0; i < input.getRows(); ++i)
                for (int j = 0; j < input.getCols(); ++j)
                    result.data[i][j] = (input.data[i][j] > 0) ? 1 : 0.01;
            return result;
        }
