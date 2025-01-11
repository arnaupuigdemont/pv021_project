#ifndef ACTIVATION_FUNCTION_HH
#define ACTIVATION_FUNCTION_HH
#include "vector.hpp"

class ActivationFunction {

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
};
#endif