#include "network.hpp"
#include "matrix.hpp"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <chrono>

// =================================================
// MLP: Entrenamiento
// =================================================
void MLP::train(const std::vector<vector> &trainData,
                const std::vector<int>    &trainLabels,
                valueType lr,
                int epochs,
                int batchSize)
{
    _lr = lr;  // antes _learningRate = learningRate

    std::vector<vector> miniBatchData(batchSize);
    std::vector<int>    miniBatchLabels(batchSize);

    int totalSamples = trainData.size();
    int batchCount   = totalSamples / batchSize;

    // Generar índices [0..totalSamples-1]
    std::vector<int> indices(totalSamples);
    for (int i = 0; i < totalSamples; ++i) {
        indices[i] = i;
    }

    for (int e = 0; e < epochs; ++e) {

        std::cout << "Epoch " << e + 1 << " / " << epochs << std::endl;

        // Barajar índices de forma determinista (puedes usar semilla variable si deseas aleatoriedad real)
        std::random_device rd;
        std::mt19937 generator(0);
        std::shuffle(indices.begin(), indices.end(), generator);

        // Iterar en batches
        for (int b = 0; b < batchCount; ++b) {
            for (int k = 0; k < batchSize; ++k) {
                int idx = indices[b * batchSize + k];
                miniBatchData[k]   = trainData[idx];
                miniBatchLabels[k] = trainLabels[idx];
            }

            // Guardamos el batch actual para el paso de update
            _trainData   = miniBatchData;   // antes _inputValues
            _trainLabels = miniBatchLabels; // antes _inputLabels

            // step = e * batchCount + b + 1
            updateWeights(e * batchCount + b + 1);
        }
    }
}

// =================================================
// MLP: Predicción
// =================================================
std::vector<int> MLP::predict(const std::vector<vector> &testData)
{
    std::vector<int> predictions;
    vector netOutput;

    for (const auto &inputVec : testData) {
        feedForward(inputVec);    // antes feedForward
        netOutput = getMLPOutput();  // antes getMLPOutput

        // Tomar argmax
        valueType maxVal = -1e9;
        int bestIndex    = 0;
        for (size_t i = 0; i < netOutput.size(); ++i) {
            if (netOutput[i] >= maxVal) {
                maxVal    = netOutput[i];
                bestIndex = i;
            }
        }
        predictions.push_back(bestIndex);
    }

    return predictions;
}

// =================================================
// MLP: Añadir capa
// =================================================
void MLP::addLayer(int outDim, ActivationType actFunc)
{
    int inDim;
    if (_layerStack.empty()) {
        inDim = _inputSize;  // antes _inputDimension
    } else {
        inDim = _layerStack.back().getDimension();
    }
    _layerStack.emplace_back(inDim, outDim, actFunc);
}

// =================================================
// MLP: Retropropagación
// =================================================
void MLP::backPropagate(size_t labelIdx)
{
    // 1) Capa de salida (one-hot)
    Layer &lastLayer = _layerStack.back();
    for (size_t i = 0; i < lastLayer.size(); ++i) {

        valueType labelOneHot = (labelIdx == i) ? 1.0 : 0.0;
        lastLayer._deltas[i]  = lastLayer._outputs[i] - labelOneHot;
    }

    // 2) Capas intermedias (propagar deltas hacia atrás)
    for (int l = static_cast<int>(_layerStack.size()) - 2; l >= 0; --l) {
        Layer &curLayer  = _layerStack[l];
        Layer &nextLayer = _layerStack[l + 1];

        for (size_t i = 0; i < curLayer.size(); ++i) {
            valueType sumDeltas = 0.0;
            for (size_t j = 0; j < nextLayer.size(); ++j) {
                sumDeltas += nextLayer._deltas[j] * nextLayer._weights[i][j];
            }
            // Multiplicar por la derivada de la activación
            curLayer._deltas[i] = sumDeltas * curLayer._valDerivs[i];
        }
    }
}

// =================================================
// MLP: Actualizar pesos
// =================================================
void MLP::updateWeights(int step)
{
    // 1) Para cada muestra en el batch => forward + backward
    for (size_t k = 0; k < _trainData.size(); ++k) {
        auto &sample = _trainData[k];
        auto  label  = _trainLabels[k];

        feedForward(sample);
        backPropagate(label);

        // Acumular gradientes (sin actualizar aún)
        for (size_t l = 0; l < _layerStack.size(); ++l) {
            Layer &layer = _layerStack[l];

            const vector &belowOutputs = (l == 0)
                                         ? sample
                                         : _layerStack[l - 1]._outputs;

            for (int i = 0; i < layer._weights.rows(); ++i) {
                for (int j = 0; j < layer._weights.cols(); ++j) {
                    layer._grads[i][j] += layer._deltas[j] * belowOutputs[i];
                }
            }
            for (size_t j = 0; j < layer.size(); ++j) {
                layer._biasGrads[j] += layer._deltas[j];
            }
        }
    }

    // 2) Aplicar Adam + limpiar gradientes
    for (auto &layer : _layerStack) {

		#pragma omp parallel for num_threads(16)
        for (int i = 0; i < layer._weights.rows(); ++i) {
            for (int j = 0; j < layer._weights.cols(); ++j) {
                // L2 regularization
                layer._grads[i][j] += regLambda * layer._weights[i][j];

                // Actualizar con Adam
                updateWeightAdam(i, j, step, layer);
				//updateWeightSGD(i, j, step, layer);
                // Limpiar grad acumulado
                layer._grads[i][j] = 0;
            }
        }

        // Bias (normalmente sin regularizar)
        for (size_t i = 0; i < layer._bias.size(); ++i) {
            updateBiasAdam(i, step, layer);
			//updateBiasSGD(i, step, layer);
            layer._biasGrads[i] = 0;
        }
    }
}

// =================================================
// MLP: Forward Pass
// =================================================
void MLP::feedForward(const vector &inputVec)
{
    vector rawZ;
    for (size_t i = 0; i < _layerStack.size(); ++i) {
        Layer &layer = _layerStack[i];

        if (i == 0) {
            // Primera capa
            rawZ = (inputVec * layer._weights) + layer._bias;
        } else {
            Layer &prevLayer = _layerStack[i - 1];
            rawZ = (prevLayer._outputs * layer._weights) + layer._bias;
        }

        // Aplicar activación
        layer._outputs    = layer.applyActivation(rawZ);
        // Guardar derivadas para backprop
        layer._valDerivs  = layer.applyActivationDeriv(rawZ);
    }
}

// =================================================
// MLP: Obtener salida final de la red
// =================================================
vector MLP::getMLPOutput()
{
    return _layerStack.back()._outputs;
}

// =================================================
// Adam en Pesos
// =================================================
void MLP::updateWeightAdam(int row, int col, int step, Layer &layer) const
{
    valueType beta1    = 0.9;
    valueType beta2    = 0.999;
    valueType eps      = 1e-8;

    valueType b1t      = std::pow(beta1, step);
    valueType b2t      = std::pow(beta2, step);

    valueType grad     = layer._grads[row][col];

    // Actualizar momentos
    layer._adamFirstMoment[row][col] =
        beta1 * layer._adamFirstMoment[row][col] + (1 - beta1) * grad;

    layer._adamSecondMoment[row][col] =
        beta2 * layer._adamSecondMoment[row][col] + (1 - beta2) * (grad * grad);

    // Corrección de sesgo
    valueType mHat = layer._adamFirstMoment[row][col] / (1 - b1t);
    valueType vHat = layer._adamSecondMoment[row][col] / (1 - b2t);

    // Actualización
    layer._weights[row][col] -= _lr * mHat / (std::sqrt(vHat) + eps);
}

// =================================================
// Adam en Bias
// =================================================
void MLP::updateBiasAdam(int idx, int step, Layer &layer) const
{
    valueType beta1 = 0.9;
    valueType beta2 = 0.999;
    valueType eps   = 1e-8;

    valueType b1t   = std::pow(beta1, step);
    valueType b2t   = std::pow(beta2, step);

    valueType grad  = layer._biasGrads[idx];

    layer._adamBiasFirstMom[idx] =
        beta1 * layer._adamBiasFirstMom[idx] + (1 - beta1) * grad;

    layer._adamBiasSecondMom[idx] =
        beta2 * layer._adamBiasSecondMom[idx] + (1 - beta2) * (grad * grad);

    valueType mHat = layer._adamBiasFirstMom[idx]     / (1 - b1t);
    valueType vHat = layer._adamBiasSecondMom[idx]    / (1 - b2t);

    layer._bias[idx] -= _lr * mHat / (std::sqrt(vHat) + eps);
}

// =================================================
// MLP: Actualizar Pesos con SGD
// =================================================
void MLP::updateWeightSGD(int row, int col, int step, Layer &layer) const {
    // Constante de momentum (puedes definirla globalmente o como atributo)
    const valueType momentum = 0.9;

    // Se asume que la regularización L2 ya se agregó al gradiente antes de llamar a esta función:
    // Es decir, el gradiente es: grad + λ * weight

    // Obtener el gradiente actual para el peso en (row, col)
    valueType grad = layer._grads[row][col];

    // Actualizar la velocidad:
    // v = momentum * v - lr * grad
    layer._sgdVelocity[row][col] = momentum * layer._sgdVelocity[row][col] - _lr * grad;

    // Actualizar el peso usando la velocidad:
    // weight = weight + v
    layer._weights[row][col] += layer._sgdVelocity[row][col];
}

void MLP::updateBiasSGD(int idx, int step, Layer &layer) const {
    const valueType momentum = 0.9;

    // Obtiene el gradiente del bias en la posición idx
    valueType grad = layer._biasGrads[idx];

    // Actualizar la velocidad del bias:
    // v_bias = momentum * v_bias - lr * grad
    layer._sgdBiasVelocity[idx] = momentum * layer._sgdBiasVelocity[idx] - _lr * grad;

    // Actualizar el bias:
    // bias = bias + v_bias
    layer._bias[idx] += layer._sgdBiasVelocity[idx];
}


// =================================================
// Layer: Usar Función de Activación
// =================================================
vector Layer::applyActivation(const vector &zVec)
{
    switch (_actType) {
        case ActivationType::LeakyReLU:
            return leakyReLu(zVec, _leakyAlpha);
        case ActivationType::Softmax:
            return softmax(zVec);
        default:
            return vector(zVec.size());
    }
}

// =================================================
// Layer: Usar Derivada de la Activación
// =================================================
vector Layer::applyActivationDeriv(const vector &zVec)
{
    switch (_actType) {
        case ActivationType::LeakyReLU:
            return leakyReLuDerivative(zVec, _leakyAlpha);
        case ActivationType::Softmax:
            return softmaxDerivative(_outputs);
        default:
            return vector(zVec.size());
    }
}

// =================================================
// MLP: Ajustar la alpha de LeakyReLU
// =================================================
void MLP::setLeakyReLUAlpha(valueType alpha)
{
    for (auto &layer : _layerStack) {
        layer._leakyAlpha = alpha;
    }
}

// =================================================
// getWeightInitByActivation + initWeights + initBias
// =================================================
WeightInitType getWeightInitByActivation(ActivationType actFunc)
{
    switch (actFunc) {
        case ActivationType::LeakyReLU:
            return WeightInitType::He;
        case ActivationType::Softmax:
        default:
            return WeightInitType::Glorot;
    }
}

matrix initWeights(int inDim, int outDim, ActivationType actFunc, bool uniformDist)
{
    matrix w(inDim, outDim);
    WeightInitType winit = getWeightInitByActivation(actFunc);

    valueType factor = (uniformDist) ? 3.0 : 1.0;
    valueType bound;
    switch (winit) {
        case WeightInitType::Glorot:
            bound = std::sqrt(factor * 2.0 / (inDim + outDim));
            break;
        case WeightInitType::He:
            bound = std::sqrt(factor * 2.0 / inDim);
            break;
    }
    valueType minVal = -bound;
    valueType maxVal =  bound;

    std::mt19937 gen(0);
    std::uniform_real_distribution<valueType> dist(minVal, maxVal);

    for (int i = 0; i < inDim; ++i) {
        for (int j = 0; j < outDim; ++j) {
            w[i][j] = dist(gen);
        }
    }
    return w;
}

vector initBias(int dim)
{
    // Inicializamos en 0
    vector b(dim);
    return b;
}
