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

        Matrix Layer::backward_output(const Matrix &grad_output, double learning_rate) {

            // Paso 1: Gradientes estándar para pesos y sesgos
            Matrix grad_weights = cached_input.transpose() * grad_output; // (input_size, batch_size) * (batch_size, output_size)
            Matrix grad_biases(1, grad_output.getCols());                 // (1, output_size)
            for (int j = 0; j < grad_output.getCols(); ++j) {
                for (int i = 0; i < grad_output.getRows(); ++i) {
                    grad_biases.data[0][j] += grad_output.data[i][j];
                }
            }

            // Paso 2: Actualización de pesos y sesgos
            weights = weights - grad_weights.scalar_mul(learning_rate);
            biases = biases - grad_biases.scalar_mul(learning_rate);

            // Paso 3: Gradiente de entrada para la siguiente capa
            Matrix grad_input = grad_output * weights.transpose(); // (batch_size, output_size) * (output_size, input_size)
            return grad_input;

        }

        Matrix Layer::backward_relu(const Matrix &grad_output, double learning_rate) {
cout << "grad_output: " << grad_output.getRows() << " " << grad_output.getCols() << endl;
            // Paso 1: Derivada de activación Leaky ReLU
            Matrix grad_activation(cached_input.getRows(), cached_input.getCols());
cout << "cached_input: " << cached_input.getRows() << " " << cached_input.getCols() << endl;
            for (int i = 0; i < cached_input.getRows(); ++i) {
                for (int j = 0; j < cached_input.getCols(); ++j) {
                    grad_activation.data[i][j] = (cached_input.data[i][j] > 0) ? grad_output.data[i][j] : 0.01 * grad_output.data[i][j];
                }
            }

            // Paso 2: Gradientes estándar para pesos y sesgos
            Matrix grad_weights = cached_input.transpose() * grad_activation; // (input_size, batch_size) * (batch_size, hidden_size)
            Matrix grad_biases(1, grad_activation.getCols());                 // (1, hidden_size)
cout << "grad_activation: " << grad_activation.getRows() << " " << grad_activation.getCols() << endl;
cout << "grad_weights: " << grad_weights.getRows() << " " << grad_weights.getCols() << endl;
cout << "grad_biases: " << grad_biases.getRows() << " " << grad_biases.getCols() << endl;
            for (int j = 0; j < grad_activation.getCols(); ++j) {
                for (int i = 0; i < grad_activation.getRows(); ++i) {
                    grad_biases.data[0][j] += grad_activation.data[i][j];
                }
            }
cout << "weights: " << weights.getRows() << " " << weights.getCols() << endl;
cout << "biases: " << biases.getRows() << " " << biases.getCols() << endl;
            // Paso 3: Actualización de pesos y sesgos
            weights = weights - grad_weights.scalar_mul(learning_rate);
            biases = biases - grad_biases.scalar_mul(learning_rate);
cout << "weights: " << weights.getRows() << " " << weights.getCols() << endl;
cout << "biases: " << biases.getRows() << " " << biases.getCols() << endl;
            // Paso 4: Gradiente de entrada para la siguiente capa
            Matrix grad_input = grad_activation * weights.transpose(); // (batch_size, hidden_size) * (hidden_size, input_size)
            return grad_input;

        }

        Matrix Layer::backward_ADAM(const Matrix &grad_output, double learning_rate, double lambda) {

           t++;

            // Paso 2: Gradientes para pesos y sesgos
            Matrix grad_weights = cached_input.transpose() * grad_output;
            Matrix grad_biases(1, grad_output.getCols());
            for (int j = 0; j < grad_output.getCols(); ++j) {
                for (int i = 0; i < grad_output.getRows(); ++i) {
                    grad_biases.data[0][j] += grad_output.data[i][j];
                }
            }

            // Paso 3: Actualización de Adam
            m_weights = m_weights.scalar_mul(beta1) + grad_weights.scalar_mul(1 - beta1);
            v_weights = v_weights.scalar_mul(beta2) + grad_weights.hadamard(grad_weights).scalar_mul(1 - beta2);
            m_biases = m_biases.scalar_mul(beta1) + grad_biases.scalar_mul(1 - beta1);
            v_biases = v_biases.scalar_mul(beta2) + grad_biases.hadamard(grad_biases).scalar_mul(1 - beta2);

            // Corrección de sesgo
            Matrix m_weights_hat = m_weights.scalar_mul(1.0 / (1.0 - pow(beta1, t)));
            Matrix v_weights_hat = v_weights.scalar_mul(1.0 / (1.0 - pow(beta2, t)));
            Matrix m_biases_hat = m_biases.scalar_mul(1.0 / (1.0 - pow(beta1, t)));
            Matrix v_biases_hat = v_biases.scalar_mul(1.0 / (1.0 - pow(beta2, t)));

            // Actualización de parámetros
            weights = weights - (m_weights_hat / (v_weights_hat.sqrt() + epsilon)).scalar_mul(learning_rate);
            biases = biases - (m_biases_hat / (v_biases_hat.sqrt() + epsilon)).scalar_mul(learning_rate);

            // Gradiente para la capa anterior
            Matrix grad_input = grad_output * weights.transpose();

            return grad_input;
        }

        Matrix Layer::backward_ADAM_relu(const Matrix &grad_output, double learning_rate, double lambda) {
            t++;

            // Paso 1: Derivada de activación
            Matrix grad_activation = cached_input.hadamard(leaky_relu_derivative(grad_output));

            // Paso 2: Gradientes para pesos y sesgos
            Matrix grad_weights = cached_input.transpose() * grad_activation;
            Matrix grad_biases(1, grad_activation.getCols());
            for (int j = 0; j < grad_activation.getCols(); ++j) {
                for (int i = 0; i < grad_activation.getRows(); ++i) {
                    grad_biases.data[0][j] += grad_activation.data[i][j];
                }
            }

            // Paso 3: Actualización de Adam
            m_weights = m_weights.scalar_mul(beta1) + grad_weights.scalar_mul(1 - beta1);
            v_weights = v_weights.scalar_mul(beta2) + grad_weights.hadamard(grad_weights).scalar_mul(1 - beta2);
            m_biases = m_biases.scalar_mul(beta1) + grad_biases.scalar_mul(1 - beta1);
            v_biases = v_biases.scalar_mul(beta2) + grad_biases.hadamard(grad_biases).scalar_mul(1 - beta2);

            // Corrección de sesgo
            Matrix m_weights_hat = m_weights.scalar_mul(1.0 / (1.0 - pow(beta1, t)));
            Matrix v_weights_hat = v_weights.scalar_mul(1.0 / (1.0 - pow(beta2, t)));
            Matrix m_biases_hat = m_biases.scalar_mul(1.0 / (1.0 - pow(beta1, t)));
            Matrix v_biases_hat = v_biases.scalar_mul(1.0 / (1.0 - pow(beta2, t)));

            // Actualización de parámetros
            weights = weights - (m_weights_hat / (v_weights_hat.sqrt() + epsilon)).scalar_mul(learning_rate);
            biases = biases - (m_biases_hat / (v_biases_hat.sqrt() + epsilon)).scalar_mul(learning_rate);

            // Gradiente para la capa anterior
            Matrix grad_input = grad_activation * weights.transpose();

            return grad_input;
        }

        Matrix Layer::backward_ADAM_output(const Matrix &grad_output, double learning_rate, double lambda) {
           
            // Incrementar el contador de pasos de Adam
            t++;

            // Gradientes de pesos y sesgos
            Matrix grad_weights = cached_input.transpose() * grad_output;
            Matrix grad_biases(1, grad_output.getCols());

            // Sumar grad_output a lo largo de las filas (optimización)
            for (int j = 0; j < grad_output.getCols(); ++j) {
                for (int i = 0; i < grad_output.getRows(); ++i) {
                    grad_biases.data[0][j] += grad_output.data[i][j];
                }
            }

            // Añadir regularización L2 al gradiente de los pesos
            grad_weights = grad_weights + weights.scalar_mul(lambda);

            // Actualización de Adam
            m_weights = m_weights.scalar_mul(beta1) + grad_weights.scalar_mul(1 - beta1);
            v_weights = v_weights.scalar_mul(beta2) + grad_weights.hadamard(grad_weights).scalar_mul(1 - beta2);
            m_biases = m_biases.scalar_mul(beta1) + grad_biases.scalar_mul(1 - beta1);
            v_biases = v_biases.scalar_mul(beta2) + grad_biases.hadamard(grad_biases).scalar_mul(1 - beta2);

            // Corrección de sesgo
            Matrix m_weights_hat = m_weights.scalar_mul(1.0 / (1.0 - pow(beta1, t)));
            Matrix v_weights_hat = v_weights.scalar_mul(1.0 / (1.0 - pow(beta2, t)));
            Matrix m_biases_hat = m_biases.scalar_mul(1.0 / (1.0 - pow(beta1, t)));
            Matrix v_biases_hat = v_biases.scalar_mul(1.0 / (1.0 - pow(beta2, t)));

            // Actualización de parámetros
            weights = weights - (m_weights_hat / (v_weights_hat.sqrt() + epsilon)).scalar_mul(learning_rate);
            biases = biases - (m_biases_hat / (v_biases_hat.sqrt() + epsilon)).scalar_mul(learning_rate);

            // Gradiente de entrada para la capa anterior
            Matrix grad_input = grad_output * weights.transpose();

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
