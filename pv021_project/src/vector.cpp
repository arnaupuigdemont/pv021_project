#include "vector.hpp"
#include <algorithm>

Vector operator+(const Vector &a, const Vector &b) {
    Vector result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

Vector operator-(const Vector &a, const Vector &b) {
    Vector result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

Vector operator*(const Vector &a, valueType scalar) {
    Vector result(a.size());
    for (int i = 0; i < a.size(); ++i) {
        result[i] = a[i] * scalar;
    }
    return result;
}

Vector operator*(valueType scalar, const Vector &a) {
    return a * scalar;
}

// Dot product operator.
valueType operator*(const Vector &a, const Vector &b) {
    valueType dotResult = 0;
    for (int i = 0; i < a.size(); ++i) {
        dotResult += a[i] * b[i];
    }
    return dotResult;
}