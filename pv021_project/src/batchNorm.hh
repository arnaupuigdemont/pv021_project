#ifndef BATCHNORM_HH
#define BATCHNORM_HH
#include "matrix.hh"
#include <vector>
#include <cmath>

class BatchNorm {
public:
    BatchNorm(int num_features, double momentum=0.9, double eps=1e-5)
        : momentum(momentum), eps(eps)
    {
        // gamma y beta se inicializan
        gamma = Matrix(1, num_features);
        beta  = Matrix(1, num_features);
        for(int j = 0; j < num_features; ++j) {
            gamma.data[0][j] = 1.0;  // Escala inicial en 1
            beta.data[0][j]  = 0.0;  // Desplazamiento inicial en 0
        }

        // moving_mean y moving_var para inference
        moving_mean = Matrix(1, num_features);
        moving_var  = Matrix(1, num_features);
    }

    // FORWARD
    Matrix forward(const Matrix &x, bool training=true) {
        // x: (batch_size, num_features)
        int batch_size = x.getRows();
        int num_features = x.getCols();

        if (mean.data.empty() || var.data.empty()) {
            // Asegurar que tengan el tamaño correcto
            mean = Matrix(1, num_features);
            var  = Matrix(1, num_features);
        }

        if (training) {
            // Calcular mean y var en el batch actual
            calculateMeanVar(x, batch_size, num_features);

            // Actualizar moving mean y moving var
            for(int j = 0; j < num_features; ++j) {
                moving_mean.data[0][j] = momentum * moving_mean.data[0][j] + 
                                         (1.0 - momentum) * mean.data[0][j];
                moving_var.data[0][j]  = momentum * moving_var.data[0][j] + 
                                         (1.0 - momentum) * var.data[0][j];
            }
        } else {
            // En inferencia usamos los promedios acumulados
            mean = moving_mean;
            var  = moving_var;
        }

        // Normalizar x
        Matrix x_norm = x;
        for(int i = 0; i < batch_size; ++i) {
            for(int j = 0; j < num_features; ++j) {
                x_norm.data[i][j] = (x.data[i][j] - mean.data[0][j]) / 
                                    std::sqrt(var.data[0][j] + eps);
            }
        }

        // Escalar y desplazar
        Matrix out = x_norm;
        for(int i = 0; i < batch_size; ++i) {
            for(int j = 0; j < num_features; ++j) {
                out.data[i][j] = gamma.data[0][j] * x_norm.data[i][j] + beta.data[0][j];
            }
        }

        // Guardar x_norm para backward
        cached_x_norm = x_norm;
        cached_input = x; // Para usar en backward
        return out;
    }

    // BACKWARD
    // Recibe grad_out: dL/d(out)
    Matrix backward(const Matrix &grad_out, double learning_rate) {
        // Referencias:
        //  dL/dx_norm = dL/d(out) * gamma
        //  dL/dvar, dL/dmean (ver fórmulas BN)

        int batch_size = grad_out.getRows();
        int num_features = grad_out.getCols();

        // 1) Gradientes w.r.t gamma y beta
        // dL/dgamma = sum( dL/dout * x_norm ), dL/dbeta = sum(dL/dout)
        for(int j = 0; j < num_features; ++j) {
            double grad_gamma = 0.0;
            double grad_beta = 0.0;
            for(int i = 0; i < batch_size; ++i) {
                grad_gamma += grad_out.data[i][j] * cached_x_norm.data[i][j];
                grad_beta  += grad_out.data[i][j];
            }
            // Actualizar gamma, beta
            gamma.data[0][j] -= learning_rate * grad_gamma;
            beta.data[0][j]  -= learning_rate * grad_beta;
        }

        // 2) dL/dx (usamos las fórmulas extendidas)
        Matrix grad_x(batch_size, num_features, 0.0);
        for(int j = 0; j < num_features; ++j) {
            // Calculamos:
            // dL/dx_norm = grad_out * gamma
            // x_norm = (x - mean) / sqrt(var + eps)
            double invStd = 1.0 / std::sqrt(var.data[0][j] + eps);

            // Suma de dL/dx_norm y x_norm*dL/dx_norm
            double sum_dout = 0.0;
            double sum_dout_xnorm = 0.0;
            for(int i = 0; i < batch_size; ++i) {
                double dOut_ij = grad_out.data[i][j];
                sum_dout       += dOut_ij;
                sum_dout_xnorm += dOut_ij * cached_x_norm.data[i][j];
            }

            // Cada x_i:
            for(int i = 0; i < batch_size; ++i) {
                // dL/dx_norm(i,j)
                double dOut_ij = grad_out.data[i][j];
                double dX_norm = dOut_ij * gamma.data[0][j];

                // Formula final:
                // dL/dx(i,j) = (1/N)*[N*dX_norm - sum(dX_norm) - x_norm(i,j)*sum(dX_norm*x_norm)] * invStd
                grad_x.data[i][j] = (dX_norm - (sum_dout / (double)batch_size) * gamma.data[0][j] * cached_x_norm.data[i][j]
                                     - (sum_dout_xnorm / (double)batch_size) * cached_x_norm.data[i][j]) 
                                    * invStd
                                    + (dX_norm - sum_dout / (double)batch_size - sum_dout_xnorm / (double)batch_size) * invStd; 
                // Nota: La fórmula exacta depende de la implementación, aquí es un ejemplo
            }
        }

        return grad_x;
    }

private:
    void calculateMeanVar(const Matrix &x, int batch_size, int num_features) {
        // Calcula mean y var
        // mean: (1, num_features)
        // var : (1, num_features)
        for(int j = 0; j < num_features; ++j) {
            double sum_val = 0.0;
            for(int i = 0; i < batch_size; ++i) {
                sum_val += x.data[i][j];
            }
            double mean_val = sum_val / batch_size;
            mean.data[0][j] = mean_val;

            // Varianza
            double sq_sum = 0.0;
            for(int i = 0; i < batch_size; ++i) {
                double diff = (x.data[i][j] - mean_val);
                sq_sum += diff * diff;
            }
            var.data[0][j] = sq_sum / batch_size;
        }
    }

public:
    Matrix gamma, beta;
    Matrix moving_mean, moving_var;

    // buffers internos
    Matrix mean, var;
    Matrix cached_x_norm;
    Matrix cached_input;

    double momentum;
    double eps;
};
#endif
