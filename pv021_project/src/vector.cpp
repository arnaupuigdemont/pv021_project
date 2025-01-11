#include "vector.hpp"
#include <algorithm>

/**
 * @brief operator* vector-vector multiplication
 */
Vector operator+(const Vector &a, const Vector &b) {
    Vector result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

/**
 * @brief operator- vector-vector subtraction
 */
Vector operator-(const Vector &a, const Vector &b) {
    Vector result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

/**
 * @brief operator* vector-scalar multiplication
 */
Vector operator*(const Vector &a, valueType scalar) {
    Vector result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] * scalar;
    }
    return result;
}

/**
 * @brief operator* scalar-vector multiplication
 */
Vector operator*(valueType scalar, const Vector &a) {
    return a * scalar;
}

/**
 * @brief operator* vector-vector dot product
 */
valueType operator*(const Vector &a, const Vector &b) {
    valueType dotResult = 0;
    for (int i = 0; i < a.size(); ++i) {
        dotResult += a[i] * b[i];
    }
    return dotResult;
}