#ifndef MATRIX_HH
#define MATRIX_HH

#include <vector>
#include <cstdint> 
#include "vector.hpp"


class Matrix {
    int _rows;
    int _cols;
    std::vector<Vector> _rowsData;

public:
    // Constructs a matrix of size rows x cols with all values initialized to 0.
    Matrix(int rows, int cols)
        : _rows(rows), _cols(cols), _rowsData(rows, Vector(cols)) {}

    // Constructs a matrix from a vector of Vectors.
    explicit Matrix(const std::vector<Vector> &rowsData)
        : _rows(rowsData.size()), _cols(rowsData[0].size()), _rowsData(rowsData) {}

    // Overloaded operators for matrix arithmetic.
    friend Matrix operator+(const Matrix &a, const Matrix &b);
    friend Vector operator*(const Matrix &m, const Vector &v);
    friend Vector operator*(const Vector &v, const Matrix &m);

    // Access operators.
    const Vector& operator[](int i) const { return _rowsData[i]; }
    Vector& operator[](int i) { return _rowsData[i]; }

    // Get a specific row.
    Vector row(int index) const;
    // Get a specific column.
    Vector col(int index) const;

    // Return the number of columns.
    int cols() const { return _cols; }
    // Return the number of rows.
    int rows() const { return _rows; }
    // Returns the number of rows.
    size_t size() const { return _rows; }

    // Return the underlying data.
    const std::vector<Vector>& getData() const { return _rowsData; }
};


// ====================================================================
// Utility Functions for Activation
// ====================================================================

// Leaky ReLU activation function applied element-wise on a single value.
valueType leakyReLu(valueType x, float alpha);

// Applies the Leaky ReLU activation on all elements of the input vector.
Vector leakyReLu(const Vector &inputVector, float alpha);

// Applies the Softmax function on the input vector.
Vector softmax(const Vector &inputVector);

// Computes the derivative of the Leaky ReLU activation (element-wise).
Vector leakyReLuDerivative(const Vector &inputVector, float alpha);

// Computes the derivative of the Softmax function.
// (Often this is computed together with the loss, but here it is defined as a separate function.)
Vector softmaxDerivative(const Vector &inputVector);

// ====================================================================
// Additional Overloaded Utility Functions
// ====================================================================

// Utility function for combining vectors with a sign factor.
// 'sign' should be either +1 or -1.
Vector plusMinusVectors(const Vector &a, const Vector &b, int sign);

#endif
