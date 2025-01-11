#include "layer.hpp"
#include <random>

// =================================================
// Layer: Usar Función de Activación
// =================================================
Vector Layer::applyActivation(const Vector &zVec) {
    return leakyReLu(zVec, _leakyAlpha);
}

Vector Layer::applyActivationOutput(const Vector &zVec) {
    return softmax(zVec);
}

// =================================================
// Layer: Usar Derivada de la Activación
// =================================================
Vector Layer::applyActivationDeriv(const Vector &zVec) {
    return leakyReLuDerivative(zVec, _leakyAlpha);
}

Vector Layer::applyActivationDerivOutput(const Vector &zVec) {
    return softmaxDerivative(zVec);
}

Matrix initWeights(int inDim, int outDim, bool output, bool uniformDist) {
    Matrix weights(inDim, outDim);
    
    valueType scaleFactor = (uniformDist) ? 3.0 : 1.0;
    valueType bound;
    
    if(output) {
        bound = std::sqrt(scaleFactor * 2.0 / (inDim + outDim));
    } else {
        bound = std::sqrt(scaleFactor * 2.0 / inDim);
    }

    valueType minVal = -bound;
    valueType maxVal = bound;

    std::mt19937 gen(0);
    std::uniform_real_distribution<valueType> dist(minVal, maxVal);

    for (int i = 0; i < inDim; ++i) {
        for (int j = 0; j < outDim; ++j) {
            weights[i][j] = dist(gen);
        }
    }
    return weights;
}


Vector initBias(int dim) {
    // Inicializamos en 0
    Vector b(dim);
    return b;
}