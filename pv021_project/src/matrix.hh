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
        Matrix operator-(const Matrix &m) const;
        Matrix operator*(const Matrix &m) const;
        Matrix transpose() const;
        Matrix scalar_mul(double scalar) const;

        static Matrix Xavier(int rows, int cols, int input_size);

        static Matrix HeIni(int rows, int cols, int input_size);

        void normalize();

        void print();

        Matrix cross_entropy_loss(const Matrix &output, const Matrix &label);

        Matrix clip_gradients(double min, double max);

    private:

};
#endif