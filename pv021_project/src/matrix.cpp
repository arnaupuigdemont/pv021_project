#include "matrix.hh"

        //CONSTRUCTOR
        Matrix::Matrix(int rows, int cols) {
            data.resize(rows, vector<double>(cols, 0.0));
        }

        Matrix::Matrix(int rows, int cols, double val) {
            data.resize(rows, vector<double>(cols, val));
        }

        Matrix Matrix::Random(int rows, int cols) {
            Matrix mat(rows, cols);
            mt19937 gen(0);
            uniform_real_distribution<> dis(-0.1, 0.1);

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    mat.data[i][j] = dis(gen);
            return mat;
        }

        Matrix::Matrix(const std::vector<std::vector<double>> &data) : data(data) {}

        int Matrix::getRows() const { return data.size(); }
        int Matrix::getCols() const { return data[0].size(); }

        //OPERATORS
        Matrix Matrix::operator+(const Matrix &other) const {
            int r = getRows(), c = getCols();
            Matrix result(r, c);
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < c; ++j)
                    result.data[i][j] = data[i][j] + other.data[i][j];
            return result;
        }

        Matrix Matrix::operator+(double scalar) const {
            Matrix result(getRows(), getCols()); 
            for (int i = 0; i < getRows(); ++i) {
                for (int j = 0; j < getCols(); ++j) {
                    result.data[i][j] = data[i][j] + scalar; 
                }
            }
            return result;
        }

        Matrix Matrix::operator-(const Matrix &other) const {
            int r = getRows(), c = getCols();
            Matrix result(r, c);
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < c; ++j)
                    result.data[i][j] = data[i][j] - other.data[i][j];
            return result;
        }

        Matrix Matrix::operator*(const Matrix &other) const {
            int r1 = getRows(), c1 = getCols();
            int c2 = other.getCols();
            Matrix result(r1, c2);
            for (int i = 0; i < r1; ++i) {
                for (int k = 0; k < c1; ++k) {
                    for (int j = 0; j < c2; ++j) {
                        result.data[i][j] += data[i][k] * other.data[k][j];
                    }
                }
            }
            return result;
        }

        Matrix Matrix::operator*(double scalar) const {
            Matrix result(getRows(), getCols());
            for (int i = 0; i < getRows(); ++i) {
                for (int j = 0; j < getCols(); ++j) {
                    result.data[i][j] = data[i][j] * scalar;
                }
            }
            return result;
        }

        Matrix Matrix::transpose() const {
            int r = getRows(), c = getCols();
            Matrix result(c, r);
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < c; ++j)
                    result.data[j][i] = data[i][j];
            return result;
        }

        Matrix Matrix::scalar_mul(double scalar) const {
            int r = getRows(), c = getCols();
            Matrix result(r, c);
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < c; ++j)
                    result.data[i][j] = data[i][j] * scalar;
            return result;
        }

        Matrix Matrix::operator/(const Matrix &other) const {
            Matrix result = *this;
            for (int i = 0; i < getRows(); ++i) {
                for (int j = 0; j < getCols(); ++j) {
                    result.data[i][j] /= other.data[i][j];
                }
            }
            return result;
        }

        Matrix Matrix::operator/(double scalar) const {

            Matrix result = *this;

            for (auto& row : result.data) {
                for (auto& elem : row) {
                    elem /= scalar;
                }
            }
            return result;
        }

        Matrix Matrix::apply_dropout(double keep_prob) {
            Matrix result(getRows(), getCols());
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);

            for (int i = 0; i < getRows(); ++i) {
                for (int j = 0; j < getCols(); ++j) {
                    result.data[i][j] = (dis(gen) < keep_prob) ? data[i][j] / keep_prob : 0.0;
                }
            }

            return result;
        }

        Matrix Matrix::broadcast_biases(const Matrix &res, const Matrix &biases) {
            if (biases.getRows() != 1 || biases.getCols() != res.getCols()) {
                throw std::invalid_argument("Bias dimensions are incompatible with result matrix");
            }

            Matrix result = res; // Copia la matriz original
            for (int i = 0; i < res.getRows(); ++i) {
                for (int j = 0; j < res.getCols(); ++j) {
                    result.data[i][j] += biases.data[0][j]; // Suma el sesgo correspondiente
                }
            }
            return result;
        }

        Matrix Matrix::Xavier(int rows, int cols, int input_size) {
            Matrix mat(rows, cols);
            double limit = std::sqrt(1.0 / input_size); 
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> dis(-limit, limit);

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    mat.data[i][j] = dis(gen);

            return mat;
        }

        Matrix Matrix::HeIni(int rows, int cols, int input_size) {
            Matrix mat(rows, cols);
            double limit = std::sqrt(2.0 / input_size); 
            random_device rd;
            mt19937 gen(0);
            uniform_real_distribution<> dis(-limit, limit);

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    mat.data[i][j] = dis(gen);

            return mat;
        }

        void Matrix::normalize() {
            int r = getRows(), c = getCols();

            for (int i = 0; i < r; ++i)
                for (int j = 0; j < c; ++j) {
                    data[i][j] /= 255.0;
                }
            
        }

        double Matrix::cross_entropy_loss(const Matrix &output, const Matrix &label) {
            double epsilon = 1e-9; // Evitar log(0)
            double loss = 0.0;

            for (int i = 0; i < output.getRows(); ++i) {
                for (int j = 0; j < output.getCols(); ++j) {
                    if (label.data[i][j] > 0) { // Solo para la clase verdadera
                        loss -= label.data[i][j] * log(output.data[i][j] + epsilon);
                    }
                }
            }

            return loss; // Promedio por el tamaño del batch
        }

        Matrix Matrix::hadamard(const Matrix &other) const {

            Matrix result(getRows(), getCols());

            for (int i = 0; i < getRows(); ++i) {
                for (int j = 0; j < getCols(); ++j) {
                    result.data[i][j] = data[i][j] * other.data[i][j];
                }
            }

            return result;
        }

        Matrix Matrix::sqrt() const {

            Matrix result(getRows(), getCols());

            for (int i = 0; i < getRows(); ++i) {
                for (int j = 0; j < getCols(); ++j) {
                    if (data[i][j] < 0) {
                        throw std::domain_error("Negative number.");
                    }
                    result.data[i][j] = std::sqrt(data[i][j]);
                }
            }

            return result;
        }

        void Matrix::print() {
            for (int i = 0; i < getRows(); ++i) {
                for (int j = 0; j < getCols(); ++j) {
                    cout << data[i][j] << " ";
                }
                cout << endl;
            }
        }

        Matrix Matrix::clip_gradients(double min_val, double max_val) {
            Matrix clipped = *this; // Create a copy of the matrix
            for (int i = 0; i < getRows(); ++i) {
                for (int j = 0; j < getCols(); ++j) {
                    clipped.data[i][j] = std::max(min_val, std::min(clipped.data[i][j], max_val));
                }
            }
            return clipped;
        }
