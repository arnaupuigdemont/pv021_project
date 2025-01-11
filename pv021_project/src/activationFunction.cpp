#include "activationFunction.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

/**
 * @brief Leaky ReLU applied on a single value.
 */
valueType activationFunction::leakyReLu(valueType x, float alpha) {
    return (x < 0) ? (x * alpha) : x;
}

/**
 * @brief Applies Leaky ReLU element-wise on a vector.
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
 * @brief Applies the derivative of Leaky ReLU element-wise on a vector.
 */
Vector activationFunction::leakyReLuDerivative(const Vector &inputVector, float alpha) {
    std::vector<valueType> data = inputVector.getData();
    std::for_each(data.begin(), data.end(),
                  [alpha](valueType &x) { x = (x < 0) ? alpha : 1.0; });
    return Vector(data);
}

/**
 * @brief Applies softmax function on a vector.
 */
Vector activationFunction::softmax(const Vector &inputVector) {
    std::vector<valueType> data = inputVector.getData();
    valueType maxVal = *std::max_element(data.begin(), data.end());

    std::vector<valueType> expValues(data.size());
    std::transform(data.begin(), data.end(), expValues.begin(),
                   [maxVal](valueType x) { return std::exp(x - maxVal); });

    valueType sumExp = std::accumulate(expValues.begin(), expValues.end(), 0.0f);

    std::vector<valueType> softmaxResult(data.size());
    std::transform(expValues.begin(), expValues.end(), softmaxResult.begin(),
                   [sumExp](valueType x) { return x / sumExp; });

    return Vector(softmaxResult);
}

/**
 * @brief Applies the derivative of softmax function on a vector.
 */
Vector activationFunction::softmaxDerivative(const Vector &inputVector) {
    std::vector<valueType> data = inputVector.getData();
    std::vector<valueType> result(data.size());

    std::transform(data.begin(), data.end(), result.begin(),
                   [](valueType x) { return x * (1 - x); });

    return Vector(result);
}