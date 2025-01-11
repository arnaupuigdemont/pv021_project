#include "layer.hpp"
#include <random>

// =================================================
// Layer: Usar Función de Activación
// =================================================
Vector Layer::applyActivation(const Vector &zVec) {
    switch (_actType) {
        case ActivationType::LeakyReLU:
            return leakyReLu(zVec, _leakyAlpha);
        case ActivationType::Softmax:
            return softmax(zVec);
        default:
            return Vector(zVec.size());
    }
}

// =================================================
// Layer: Usar Derivada de la Activación
// =================================================
Vector Layer::applyActivationDeriv(const Vector &zVec) {
    switch (_actType) {
        case ActivationType::LeakyReLU:
            return leakyReLuDerivative(zVec, _leakyAlpha);
        case ActivationType::Softmax:
            return softmaxDerivative(_outputs);
        default:
            return Vector(zVec.size());
    }
}

// =================================================
// getWeightInitByActivation + initWeights + initBias
// =================================================
WeightInitType getWeightInitByActivation(ActivationType actFunc) {
    switch (actFunc) {
        case ActivationType::LeakyReLU:
            return WeightInitType::He;
        case ActivationType::Softmax:
        default:
            return WeightInitType::Glorot;
    }
}

Matrix initWeights(int inDim, int outDim, ActivationType actFunc, bool uniformDist) {
    Matrix weights(inDim, outDim);
    WeightInitType initType = getWeightInitByActivation(actFunc);

    valueType scaleFactor = (uniformDist) ? 3.0 : 1.0;
    valueType bound;
    switch (initType) {
        case WeightInitType::Glorot:
            bound = std::sqrt(scaleFactor * 2.0 / (inDim + outDim));
            break;
        case WeightInitType::He:
            bound = std::sqrt(scaleFactor * 2.0 / inDim);
            break;
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