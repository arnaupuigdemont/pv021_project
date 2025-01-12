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

    static valueType sigmoid(valueType x);
    static Vector sigmoid(const Vector &inputVector);
    static Vector sigmoidDerivative(const Vector &inputVector);

    static valueType tanh(valueType x);
    static Vector tanh(const Vector &inputVector);
    static Vector tanhDerivative(const Vector &inputVector);

    static valueType relu(valueType x);
    static Vector relu(const Vector &inputVector);
    static Vector reluDerivative(const Vector &inputVector);
};

#endif // ACTIVATIONFUNCTION_HPP