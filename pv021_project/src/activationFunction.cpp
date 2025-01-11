#include "activationFunction.hpp"

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
valueType activationFunction::leakyReLu(valueType x, float alpha) {
    return (x < 0) ? (x * alpha) : x;
}

/**
 * @brief Applies Leaky ReLU element-wise on a vector.
 *
 * @param inputVector Vector of input values.
 * @param alpha Slope for negative values.
 * @return Vector Activated vector.
 */
Vector activationFunction::leakyReLu(const Vector &inputVector, float alpha) {
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
Vector activationFunction::leakyReLuDerivative(const Vector &inputVector, float alpha) {
    int dim = inputVector.size();
    std::vector<valueType> derivatives(dim);
    
    const std::vector<valueType>& data = inputVector.getData();
    std::for_each(data.begin(), data.end(), 
        [&, idx = 0](valueType x) mutable {
            derivatives[idx] = (x <= 0) ? alpha : 1.0;
            ++idx;
        });
    
    return Vector(derivatives);
}

/**
 * @brief Computes the softmax of an input vector.
 *
 * @param inputVector Input vector.
 * @return Vector Softmax probabilities.
 */
Vector activationFunction::softmax(const Vector &inputVector) {
    int dim = inputVector.size();
    const std::vector<valueType>& data = inputVector.getData();
    
    // Find the maximum element (for numerical stability)
    valueType maxVal = *std::max_element(data.begin(), data.end());
    
    // Compute exponentials in one pass using transform
    std::vector<valueType> expValues(dim);
    std::transform(data.begin(), data.end(), expValues.begin(),
                   [maxVal](valueType x) { return std::exp(x - maxVal); });
    
    // Sum the exponential values using accumulate
    valueType sumExp = std::accumulate(expValues.begin(), expValues.end(), 0.0f);
    
    // Normalize each exponential by the sum
    std::vector<valueType> softmaxResult(dim);
    std::transform(expValues.begin(), expValues.end(), softmaxResult.begin(),
                   [sumExp](valueType x) { return x / sumExp; });
    
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
Vector activationFunction::softmaxDerivative(const Vector &inputVector) {
    int dim = inputVector.size();
    std::vector<valueType> deriv(dim);
    const std::vector<valueType>& data = inputVector.getData();
    for (int i = 0; i < dim; ++i) {
        deriv[i] = data[i] * (1 - data[i]);
    }
    return Vector(deriv);
}