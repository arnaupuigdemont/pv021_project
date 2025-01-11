#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <vector>

using valueType = float;

class Vector {
    int _size;
    std::vector<valueType> _data;

public:
    // Default constructor.
    Vector() = default; 

    // Constructs a vector of given size, with all elements initialized to 0.
    explicit Vector(int size)
        : _size(size), _data(size, 0) {}

    // Constructs a vector from a standard vector.
    explicit Vector(const std::vector<valueType> &data)
        : _size(data.size()), _data(data) {}

    // Overloaded operators for vector arithmetic.
    friend Vector operator+(const Vector &a, const Vector &b);
    friend Vector operator-(const Vector &a, const Vector &b);
    friend Vector operator*(const Vector &a, valueType scalar);
    friend Vector operator*(valueType scalar, const Vector &a);
    friend valueType operator*(const Vector &a, const Vector &b); // dot product

    // Access operators.
    valueType operator[](int i) const { return _data[i]; }
    valueType &operator[](int i) { return _data[i]; }

    // Returns the vector size.
    int size() const { return _size; }

    // Some utility methods.
    void pop_back() { _data.pop_back(); }
    valueType back() const { return _data.back(); }
    void emplace_back(valueType value) { _data.emplace_back(value); }
    const std::vector<valueType>& getData() const { return _data; }
};
Vector plusMinusVectors(const Vector &a, const Vector &b, int sign);

#endif