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
                int batchSize) {
    // Establecer la tasa de aprendizaje global
    _lr = lr;
    
    int totalSamples = trainData.size();
    int numBatches = totalSamples / batchSize;
    
    // Generar vector de índices [0, 1, ..., totalSamples-1]
    std::vector<int> indices(totalSamples);
    for (int i = 0; i < totalSamples; ++i) {
        indices[i] = i;
    }
    
    // Iterar sobre las épocas
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << " / " << epochs << std::endl;
        
        // Mezclar índices de forma determinista (usar semilla fija para reproducibilidad)
        std::mt19937 rng(0);
        std::shuffle(indices.begin(), indices.end(), rng);
        
        // Procesar cada mini-batch
        for (int batch = 0; batch < numBatches; ++batch) {
            // Crear mini-batch para datos y etiquetas
            std::vector<vector> currentBatchData(batchSize);
            std::vector<int>    currentBatchLabels(batchSize);
            
            for (int i = 0; i < batchSize; ++i) {
                int dataIndex = indices[batch * batchSize + i];
                currentBatchData[i]   = trainData[dataIndex];
                currentBatchLabels[i] = trainLabels[dataIndex];
            }
            
            // Almacenar el batch actual en las variables internas del MLP
            _trainData   = currentBatchData;
            _trainLabels = currentBatchLabels;
            
            // Calcular el "globalStep" (por ejemplo, para actualizar gradientes)
            int globalStep = epoch * numBatches + batch + 1;
            
            // Actualizar pesos usando los datos acumulados del batch
            updateWeights(globalStep);
        }
    }
}


// =================================================
// MLP: Predicción
// =================================================
std::vector<int> MLP::predict(const std::vector<vector> &testData) {
    std::vector<int> predictedLabels;

    // Para cada muestra en el conjunto de test
    for (const auto &sample : testData) {
        // Realiza el forward pass con la muestra
        feedForward(sample);

        // Recupera la salida final de la red
        vector outputVector = getMLPOutput();

        // Buscar el índice del elemento de mayor valor (argmax)
        valueType currentMax = -1e9;
        int predictedIndex = 0;
        for (size_t i = 0; i < outputVector.size(); ++i) {
            if (outputVector[i] > currentMax) {
                currentMax = outputVector[i];
                predictedIndex = static_cast<int>(i);
            }
        }
        predictedLabels.push_back(predictedIndex);
    }
    
    return predictedLabels;
}


// =================================================
// MLP: Añadir capa
// =================================================
void MLP::addLayer(int outDim, ActivationType actFunc) {
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
void MLP::backPropagate(size_t targetLabel) {
    // Paso 1: Calcular el error en la capa de salida (codificación one‑hot)
    Layer &outputLayer = _layerStack.back();
    for (size_t i = 0; i < outputLayer.size(); ++i) {
        // Si el índice coincide con targetLabel, se espera 1.0; en caso contrario, 0.0.
        valueType expected = (targetLabel == i) ? 1.0 : 0.0;
        // La delta es la diferencia entre la salida real y la esperada.
        outputLayer._deltas[i] = outputLayer._outputs[i] - expected;
    }

    // Paso 2: Retropropagar el error a través de las capas ocultas
    // Se recorre desde la penúltima capa hasta la primera (en orden inverso)
    for (int l = static_cast<int>(_layerStack.size()) - 2; l >= 0; --l) {
        Layer &currentLayer = _layerStack[l];
        Layer &nextLayer    = _layerStack[l + 1];

        for (size_t i = 0; i < currentLayer.size(); ++i) {
            valueType deltaSum = 0.0;
            // Sumar la contribución de los deltas de la siguiente capa ponderados por sus pesos
            for (size_t j = 0; j < nextLayer.size(); ++j) {
                deltaSum += nextLayer._deltas[j] * nextLayer._weights[i][j];
            }
            // Multiplicar la suma de deltas por la derivada de la activación de la neurona actual
            currentLayer._deltas[i] = deltaSum * currentLayer._valDerivs[i];
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
void MLP::feedForward(const vector &inputVec) {
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
vector MLP::getMLPOutput() {
    return _layerStack.back()._outputs;
}

// =================================================
// Adam en Pesos
// =================================================
void MLP::updateWeightAdam(int row, int col, int step, Layer &layer) const {
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
void MLP::updateBiasAdam(int idx, int step, Layer &layer) const {
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
vector Layer::applyActivation(const vector &zVec) {
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
vector Layer::applyActivationDeriv(const vector &zVec) {
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
void MLP::setLeakyReLUAlpha(valueType alpha) {
    for (auto &layer : _layerStack) {
        layer._leakyAlpha = alpha;
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

matrix initWeights(int inDim, int outDim, ActivationType actFunc, bool uniformDist) {
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

vector initBias(int dim) {
    // Inicializamos en 0
    vector b(dim);
    return b;
}
