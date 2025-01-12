#include <cmath>
#include <limits>
#include <iostream>
#include <numeric>
#include <algorithm>
#include "matrix.hpp"
#include "activationFunction.hpp"
#include "vector.hpp"
#include <omp.h>


/**
 * @brief operator+ sum of two vectors
 */
Matrix operator+(const Matrix &A, const Matrix &B) {
    Matrix sum(A.rows(), A.cols());
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            sum[i][j] = A[i][j] + B[i][j];
        }
    }
    return sum;
}

/**
 * @brief operator* matrix-vector multiplication
 */
Vector operator*(const Matrix &m, const Vector &v) {
    Vector result(m.rows());
    for (int i = 0; i < m.rows(); ++i) {
        result[i] = m.row(i) * v;
    }
    return result;
}

/**
 * @brief operator* vector-matrix multiplication
 */
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

/**
 * @brief operator* matrix-matrix multiplication
 */
Matrix operator*(const Matrix &A, const Matrix &B) {
    Matrix result(A.rows(), B.cols());
    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < B.cols(); ++j) {
            result[i][j] = A.row(i) * B.col(j);
        }
    }
    return result;
}

/**
 * @brief operator* matrix-scalar multiplication
 */
Matrix operator*(const Matrix &m, valueType scalar) {
    Matrix result(m.rows(), m.cols());
    for (int i = 0; i < m.rows(); ++i) {
        for (int j = 0; j < m.cols(); ++j) {
            result[i][j] = m[i][j] * scalar;
        }
    }
    return result;
}

/**
 * @brief operator* scalar-matrix multiplication
 */
Matrix operator*(valueType scalar, const Matrix &m) {
    return m * scalar;
}

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


