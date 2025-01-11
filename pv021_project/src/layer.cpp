#include "layer.hpp"
#include "activationFunction.hpp" // Incluir el archivo de cabecera adecuado
#include <random>

/**
 * @brief apply activation function to the input vector
 */
Vector Layer::applyActivation(const Vector &zVec) {
    return activationFunction::leakyReLu(zVec, _leakyAlpha);
}

/**
 * @brief apply activation function to the output vector
 */
Vector Layer::applyActivationOutput(const Vector &zVec) {
    return activationFunction::softmax(zVec);
}

/**
 * @brief apply derivative of activation function to the input vector
 */
Vector Layer::applyActivationDeriv(const Vector &zVec) {
    return activationFunction::leakyReLuDerivative(zVec, _leakyAlpha);
}

/**
 * @brief apply derivative of activation function to the output vector
 */
Vector Layer::applyActivationDerivOutput(const Vector &zVec) {
    return activationFunction::softmaxDerivative(zVec);
}

/**
 * @brief Initialize weights for the layer
 */
Matrix initWeights(int inDim, int outDim, bool output, bool uniformDist) {
    Matrix weights(inDim, outDim);
    
    valueType scaleFactor = (uniformDist) ? 3.0 : 1.0;
    valueType bound;
    
    if(output) {
        bound = std::sqrt(scaleFactor * 2.0 / (inDim + outDim));
    } else {
        bound = std::sqrt(scaleFactor * 2.0 / inDim);
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<valueType> dist(-bound, bound);

    for (int i = 0; i < inDim; ++i) {
        for (int j = 0; j < outDim; ++j) {
            weights[i][j] = dist(gen);
        }
    }

    return weights;
}

/**
 * @brief Initialize bias for the layer
 */
Vector initBias(int dim) {
    // Inicializamos en 0
    Vector b(dim);
    return b;
}