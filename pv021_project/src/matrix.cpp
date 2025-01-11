#include <cmath>
#include <limits>
#include <iostream>
#include <numeric>
#include <algorithm>
#include "matrix.hpp"
#include "activationFunction.hpp"
#include "vector.hpp"
#include <omp.h>


// ====================================================================
// Overloaded Operators for Matrix
// ====================================================================
Matrix operator+(const Matrix &A, const Matrix &B) {
    Matrix sum(A.rows(), A.cols());
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            sum[i][j] = A[i][j] + B[i][j];
        }
    }
    return sum;
}

Vector operator*(const Matrix &m, const Vector &v) {
    Vector result(m.rows());
    for (int i = 0; i < m.rows(); ++i) {
        result[i] = m.row(i) * v;
    }
    return result;
}

Vector operator*(const Vector &v, const Matrix &m) {
    Vector result(m.cols());
#pragma omp parallel for num_threads(16)
    for (int j = 0; j < m.cols(); ++j) {
        // Multiplicamos el vector v por la columna j de la matriz m
        // asumiendo que la función col(j) devuelve un Vector
        result[j] = v * m.col(j);
    }
    return result;
}

// ====================================================================
// Matrix Row and Column Setters
// ====================================================================
Vector Matrix::row(int i) const {
    Vector r(cols());
    for (int j = 0; j < cols(); ++j) {
        r[j] = (*this)[i][j];
    }
    return r;
}

Vector Matrix::col(int j) const {
    Vector c(rows());
    for (int i = 0; i < rows(); ++i) {
        c[i] = (*this)[i][j];
    }
    return c;
}

 // Return the number of columns.
    int Matrix::cols() const { return _cols; }
    // Return the number of rows.
    int Matrix::rows() const { return _rows; }
    // Returns the number of rows.
    size_t Matrix::size() const { return _rows; }

    const std::vector<Vector>& Matrix::getData() const { return _rowsData; }


