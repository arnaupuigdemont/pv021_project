#ifndef ACTIVATIONFUNCTION_HPP
#define ACTIVATIONFUNCTION_HPP

#include "vector.hpp"

class activationFunction {
public:
    static valueType leakyReLu(valueType x, float alpha);
    static Vector leakyReLu(const Vector &inputVector, float alpha);
    static Vector leakyReLuDerivative(const Vector &inputVector, float alpha);
    static Vector softmax(const Vector &inputVector);
    static Vector softmaxDerivative(const Vector &inputVector);
};

#endif // ACTIVATIONFUNCTION_HPP