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
void MLP::train(const std::vector<Vector> &trainData,
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
            std::vector<Vector> currentBatchData(batchSize);
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
std::vector<int> MLP::predict(const std::vector<Vector> &testData) {
    std::vector<int> predictedLabels;

    // Para cada muestra en el conjunto de test
    for (const auto &sample : testData) {
        // Realiza el forward pass con la muestra
        feedForward(sample);

        // Recupera la salida final de la red
        Vector outputVector = getMLPOutput();

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
void MLP::updateWeights(int globalStep) {
    // Bloque 1: Acumular gradientes de cada muestra en el batch
    for (size_t sampleIdx = 0; sampleIdx < _trainData.size(); ++sampleIdx) {
        auto &sampleInput = _trainData[sampleIdx];
        auto  sampleLabel = _trainLabels[sampleIdx];

        // Realizar propagación hacia adelante y calcular el error
        feedForward(sampleInput);
        backPropagate(sampleLabel);

        // Propagar y acumular gradientes en cada capa
        for (size_t layerIdx = 0; layerIdx < _layerStack.size(); ++layerIdx) {
            Layer &currentLayer = _layerStack[layerIdx];

            // Si es la primera capa, la entrada es la muestra; 
            // en caso contrario, se usa la salida de la capa anterior.
            const Vector &inputForLayer = (layerIdx == 0)
                                          ? sampleInput
                                          : _layerStack[layerIdx - 1]._outputs;

            // Acumular gradientes para los pesos
            for (int i = 0; i < currentLayer._weights.rows(); ++i) {
                for (int j = 0; j < currentLayer._weights.cols(); ++j) {
                    currentLayer._grads[i][j] += currentLayer._deltas[j] * inputForLayer[i];
                }
            }
            // Acumular gradientes para los biases
            for (size_t j = 0; j < currentLayer.size(); ++j) {
                currentLayer._biasGrads[j] += currentLayer._deltas[j];
            }
        }
    }

    // Bloque 2: Actualizar parámetros (pesos y biases) de cada capa
    for (auto &layer : _layerStack) {
        // Actualizar pesos
        #pragma omp parallel for num_threads(16)
        for (int i = 0; i < layer._weights.rows(); ++i) {
            for (int j = 0; j < layer._weights.cols(); ++j) {
                // Añadir regularización L2: gradiente += regLambda * peso
                layer._grads[i][j] += regLambda * layer._weights[i][j];

                // Actualizar el peso usando SGD con momentum (o Adam, según el método seleccionado)
                // Aquí se usa la función updateWeightSGD (o Adam, comentada) con la versión reordenada.
                //updateWeightSGD(i, j, globalStep, layer);
				updateWeightAdam(i, j, globalStep, layer);
                // Reiniciar el gradiente acumulado
                layer._grads[i][j] = 0;
            }
        }

        // Actualizar biases (usualmente sin regularización)
        for (size_t i = 0; i < layer._bias.size(); ++i) {
            //updateBiasSGD(i, globalStep, layer);
			updateBiasAdam(i, globalStep, layer);
            layer._biasGrads[i] = 0;
        }
    }
}


// =================================================
// MLP: Forward Pass
// =================================================
void MLP::feedForward(const Vector &inputVec) {
    Vector preActivation;  // 'rawZ': salida de la operación lineal

    // Recorrer cada capa en el stack
    for (size_t layerIndex = 0; layerIndex < _layerStack.size(); ++layerIndex) {
        Layer &currentLayer = _layerStack[layerIndex];

        // Calcular la preactivación: z = X * W + b
        if (layerIndex == 0) {
            preActivation = (inputVec * currentLayer._weights) + currentLayer._bias;
        } else {
            Layer &prevLayer = _layerStack[layerIndex - 1];
            preActivation = (prevLayer._outputs * currentLayer._weights) + currentLayer._bias;
        }

        // Aplicar la activación y guardar resultados
        currentLayer._outputs   = currentLayer.applyActivation(preActivation);
        currentLayer._valDerivs = currentLayer.applyActivationDeriv(preActivation);
    }
}


// =================================================
// MLP: Obtener salida final de la red
// =================================================
Vector MLP::getMLPOutput() {
    return _layerStack.back()._outputs;
}

// =================================================
// Adam en Pesos
// =================================================
void MLP::updateWeightAdam(int row, int col, int step, Layer &layer) const {
    // Constantes del método Adam
    const valueType beta1 = 0.9;
    const valueType beta2 = 0.999;
    const valueType eps   = 1e-8;

    // 1. Extraer el gradiente actual del peso en (row, col)
    valueType grad = layer._grads[row][col];

    // 2. Calcular los factores de corrección de sesgo (dependientes del paso)
    valueType b1Correction = 1 - std::pow(beta1, step);
    valueType b2Correction = 1 - std::pow(beta2, step);

    // 3. Actualizar los momentos acumulados para este peso
    layer._adamFirstMoment[row][col] = beta1 * layer._adamFirstMoment[row][col] + (1 - beta1) * grad;
    layer._adamSecondMoment[row][col] = beta2 * layer._adamSecondMoment[row][col] + (1 - beta2) * (grad * grad);

    // 4. Calcular los momentos corregidos (mHat y vHat)
    valueType mHat = layer._adamFirstMoment[row][col] / b1Correction;
    valueType vHat = layer._adamSecondMoment[row][col] / b2Correction;

    // 5. Actualizar el peso: restamos la corrección escalada
    layer._weights[row][col] -= _lr * mHat / (std::sqrt(vHat) + eps);
}


// =================================================
// Adam en Bias
// =================================================
void MLP::updateBiasAdam(int idx, int step, Layer &layer) const {
    // Constantes de Adam para el bias
    const valueType beta1 = 0.9;
    const valueType beta2 = 0.999;
    const valueType eps   = 1e-8;

    // 1. Obtener el gradiente del bias en la posición idx
    valueType grad = layer._biasGrads[idx];

    // 2. Calcular los factores de corrección de sesgo
    valueType b1Correction = 1 - std::pow(beta1, step);
    valueType b2Correction = 1 - std::pow(beta2, step);

    // 3. Actualizar los momentos del bias
    layer._adamBiasFirstMom[idx]  = beta1 * layer._adamBiasFirstMom[idx]  + (1 - beta1) * grad;
    layer._adamBiasSecondMom[idx] = beta2 * layer._adamBiasSecondMom[idx] + (1 - beta2) * (grad * grad);

    // 4. Calcular los momentos corregidos
    valueType mHat = layer._adamBiasFirstMom[idx]  / b1Correction;
    valueType vHat = layer._adamBiasSecondMom[idx] / b2Correction;

    // 5. Actualizar el bias: restar el término corregido
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

Matrix initWeights(int inDim, int outDim, ActivationType actFunc, bool uniformDist = true) {
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
