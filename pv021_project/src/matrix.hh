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

        void normalize();

        Matrix cross_entropy_loss(const Matrix &output, const Matrix &label);

    private:

};
#endif