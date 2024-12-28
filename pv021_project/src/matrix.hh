#ifndef MATRIX_HH
#define MATRIX_HH

#include <vector>
#include <iostream>
#include <random>
#include <algorithm>

using namespace std;

class Matrix {
    
    public:

        vector<vector<double>> data;

        //CONSTRUCTOR
        Matrix(int rows, int cols);  
        Matrix(int rows, int cols, double val);
        static Matrix Random(int rows, int cols);

        Matrix(const std::vector<std::vector<double>> &data);

        int getRows() const;
        int getCols() const;

        //OPERATORS
        Matrix operator+(const Matrix &m) const;
        Matrix operator+(double scalar) const;
        Matrix operator-(const Matrix &m) const;
        Matrix operator*(const Matrix &m) const;
        Matrix operator/(const Matrix &m) const;
        Matrix operator/(double scalar) const;
        Matrix transpose() const;
        Matrix scalar_mul(double scalar) const;

        Matrix apply_dropout(double keep_prob);
        Matrix broadcast_biases(const Matrix &res, const Matrix &biases);

        static Matrix Xavier(int rows, int cols, int input_size);

        static Matrix HeIni(int rows, int cols, int input_size);

        void normalize();

        Matrix hadamard(const Matrix &m) const;

        Matrix sqrt() const;

        void print();

        double cross_entropy_loss(const Matrix &output, const Matrix &label);

        Matrix clip_gradients(double min, double max);

    private:

};
#endif