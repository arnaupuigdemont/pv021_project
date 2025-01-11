#include <cmath>
#include <limits>
#include <iostream>
#include "matrix.hpp"



// ====================================================================
// Overloaded Operators for Matrix
// ====================================================================
Matrix operator+(const Matrix &A, const Matrix &B) {
    int rows = A.rows();
    int cols = A.cols();
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result[i][j] = A[i][j] + B[i][j];
    return result;
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
    for (int i = 0; i < m.cols(); ++i) {
        result[i] = v * m.col(i);
    }
    return result;
}

// ====================================================================
// Matrix Row and Column Getters
// ====================================================================
Vector Matrix::row(int index) const {
    return _rowsData[index];
}

Vector Matrix::col(int index) const {
    Vector column(_rows);
    for (int i = 0; i < _rows; ++i) {
        column[i] = _rowsData[i][index];
    }
    return column;
}

// ====================================================================
// Activation Functions
// ====================================================================

/**
 * @brief Leaky ReLU applied on a single value.
 *
 * @param x Input value.
 * @param alpha Slope for negative values.
 * @return valueType Activated value.
 */
valueType leakyReLu(valueType x, float alpha) {
    return (x < 0) ? (x * alpha) : x;
}

/**
 * @brief Applies Leaky ReLU element-wise on a vector.
 *
 * @param inputVector Vector of input values.
 * @param alpha Slope for negative values.
 * @return Vector Activated vector.
 */
Vector leakyReLu(const Vector &inputVector, float alpha) {
    int dim = inputVector.size();
    std::vector<valueType> activated(dim);
    for (int i = 0; i < dim; ++i) {
        activated[i] = leakyReLu(inputVector[i], alpha);
    }
    return Vector(activated);
}

/**
 * @brief Computes the element-wise derivative of the Leaky ReLU function.
 *
 * @param inputVector Vector of input values.
 * @param alpha Slope for negative values.
 * @return Vector Derivative vector.
 */
Vector leakyReLuDerivative(const Vector &inputVector, float alpha) {
    int dim = inputVector.size();
    std::vector<valueType> derivative(dim);
    for (int i = 0; i < dim; ++i) {
        derivative[i] = (inputVector[i] <= 0) ? alpha : 1.0;
    }
    return Vector(derivative);
}

/**
 * @brief Computes the softmax of an input vector.
 *
 * @param inputVector Input vector.
 * @return Vector Softmax probabilities.
 */
Vector softmax(const Vector &inputVector) {
    int dim = inputVector.size();
    valueType maxValue = -std::numeric_limits<valueType>::infinity();
    // Find the maximum value (for numerical stability)
    for (int i = 0; i < dim; ++i) {
        if (inputVector[i] > maxValue) {
            maxValue = inputVector[i];
        }
    }

    // Compute exponential values and their sum
    valueType expSum = 0.0;
    std::vector<valueType> expValues(dim);
    for (int i = 0; i < dim; ++i) {
        expValues[i] = std::exp(inputVector[i] - maxValue);
        expSum += expValues[i];
    }

    // Normalize to get softmax outputs.
    std::vector<valueType> softmaxResult(dim);
    for (int i = 0; i < dim; ++i) {
        softmaxResult[i] = expValues[i] / expSum;
    }
    return Vector(softmaxResult);
}

/**
 * @brief Computes an approximate element-wise derivative of the softmax function.
 *
 * Note: Often computed jointly with the cross entropy loss.
 *
 * @param inputVector Vector containing softmax outputs.
 * @return Vector Element-wise derivative values.
 */
Vector softmaxDerivative(const Vector &inputVector) {
    int dim = inputVector.size();
    std::vector<valueType> deriv(dim);
    const std::vector<valueType>& data = inputVector.getData();
    for (int i = 0; i < dim; ++i) {
        deriv[i] = data[i] * (1 - data[i]);
    }
    return Vector(deriv);
}


