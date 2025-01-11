#include "vector.hpp"
#include <algorithm>

Vector operator+(const Vector &a, const Vector &b) {
    return plusMinusVectors(a, b, 1);
}

Vector operator-(const Vector &a, const Vector &b) {
    return plusMinusVectors(a, b, -1);
}

Vector operator*(const Vector &a, valueType scalar) {
    int dim = a.size();
    std::vector<valueType> newValues(dim);
    for (int i = 0; i < dim; ++i) {
        newValues[i] = a[i] * scalar;
    }
    return Vector(newValues);
}

Vector operator*(valueType scalar, const Vector &a) {
    return a * scalar;
}

// Dot product operator.
valueType operator*(const Vector &a, const Vector &b) {
    int dim = a.size();
    valueType dotProduct = 0;
    for (int i = 0; i < dim; ++i) {
        dotProduct += a[i] * b[i];
    }
    return dotProduct;
}

Vector plusMinusVectors(const Vector &a, const Vector &b, int sign) {
    int dim = a.size();
    std::vector<valueType> result(dim);
    
    const std::vector<valueType>& aData = a.getData();
    const std::vector<valueType>& bData = b.getData();
    
    std::transform(aData.begin(), aData.end(), bData.begin(), result.begin(),
                   [sign](valueType aElem, valueType bElem) {
                       return aElem + sign * bElem;
                   });
    return Vector(result);
}