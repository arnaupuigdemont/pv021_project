#include "matrix.hh"

class Matrix {
    
    
    public:

        vector<vector<double>> data;

        //CONSTRUCTOR
        Matrix::Matrix(int rows, int cols) {
            data.resize(rows, vector<double>(cols, 0.0));
        }

        static Matrix Random(int rows, int cols) {
            Matrix mat(rows, cols);
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> dis(-0.1, 0.1);

            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    mat.data[i][j] = dis(gen);
            return mat;
        }

        Matrix(const std::vector<std::vector<double>> &data) : data(data) {}

        int getRows() const { return data.size(); }
        int getCols() const { return data[0].size(); }

        //OPERATORS
        Matrix operator+(const Matrix &other) const {
            int r = getRows(), c = getCols();
            Matrix result(r, c);
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < c; ++j)
                    result.data[i][j] = data[i][j] + other.data[i][j];
            return result;
        }

        Matrix operator-(const Matrix &other) const {
            int r = getRows(), c = getCols();
            Matrix result(r, c);
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < c; ++j)
                    result.data[i][j] = data[i][j] - other.data[i][j];
            return result;
        }

        Matrix operator*(const Matrix &other) const {
            int r1 = getRows(), c1 = getCols();
            int r2 = other.getRows(), c2 = other.getCols();
            Matrix result(r1, c2);
            for (int i = 0; i < r1; ++i)
                for (int j = 0; j < c2; ++j)
                    for (int k = 0; k < c1; ++k)
                        result.data[i][j] += data[i][k] * other.data[k][j];
            return result;
        }

        Matrix transpose() const {
            int r = getRows(), c = getCols();
            Matrix result(c, r);
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < c; ++j)
                    result.data[j][i] = data[i][j];
            return result;
        }

        Matrix scalar_mul(double scalar) const {
            int r = getRows(), c = getCols();
            Matrix result(r, c);
            for (int i = 0; i < r; ++i)
                for (int j = 0; j < c; ++j)
                    result.data[i][j] = data[i][j] * scalar;
            return result;
        }

        void normalize() {
            int r = getRows(), c = getCols();

            for (int i = 0; i < r; ++i)
                for (int j = 0; j < c; ++j) {
                    data[i][j] /= 255.0;
                }
            
        }

        Matrix cross_entropy_loss(const Matrix &output, const Matrix &label) {
            Matrix loss(1, output.getCols());
            for (int i = 0; i < output.getCols(); ++i) {
                loss.data[0][i] = -label.data[0][i] * log(output.data[0][i] + 1e-9);
            }
            return loss;
        }
};