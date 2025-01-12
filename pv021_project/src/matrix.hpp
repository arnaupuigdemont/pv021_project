#ifndef MATRIX_HH
#define MATRIX_HH

#include <vector>
#include <cstdint> 
#include "vector.hpp"
#include "activationFunction.hpp"


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

    // Constructs a matrix from a vector of vectors.
    explicit Matrix(const std::vector<std::vector<valueType>> &data)
    : _rows(data.size()), _cols(data[0].size()), _rowsData(data.size()) {
        for (int i = 0; i < _rows; ++i) {
            _rowsData[i] = Vector(data[i]);
        }
    }

    // Constructs a matrix from a single vector.
    explicit Matrix(const Vector &v)
    : _rows(v.size()), _cols(1), _rowsData(1, v) {}

    // Overloaded operators for matrix arithmetic.
    friend Matrix operator+(const Matrix &a, const Matrix &b);
    friend Vector operator*(const Matrix &m, const Vector &v);
    friend Vector operator*(const Vector &v, const Matrix &m);
    friend Matrix operator*(const Matrix &a, const Matrix &b);
    friend Matrix operator*(const Matrix &a, valueType scalar);
    friend Matrix operator*(valueType scalar, const Matrix &a);

    // Access operators.
    const Vector& operator[](int i) const { return _rowsData[i]; }
    Vector& operator[](int i) { return _rowsData[i]; }

    // Get a specific row.
    Vector row(int index) const;
    // Get a specific column.
    Vector col(int index) const;

    // Return the number of columns.
    int cols() const;
    // Return the number of rows.
    int rows() const;
    // Returns the number of rows.
    size_t size() const;

    // Return the underlying data.
    const std::vector<Vector>& getData() const;
};

#endif
