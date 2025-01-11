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

    // For each sample in the test set:
    for (const auto &sample : testData) {
        // Perform forward pass with the sample.
        feedForward(sample);

        // Retrieve the final output of the network.
        Vector outputVector = getMLPOutput();
        const std::vector<valueType>& outputData = outputVector.getData();

        // Find the index of the maximum element using std::max_element.
        // We use a lambda to compare elements.
        auto maxIt = std::max_element(outputData.begin(), outputData.end());
        
        // Calculate the index as the distance from the beginning.
        int predictedIndex = static_cast<int>(std::distance(outputData.begin(), maxIt));

        predictedLabels.push_back(predictedIndex);
    }

    return predictedLabels;
}


// =================================================
// MLP: Añadir capa
// =================================================
void MLP::addLayer(int outDim) {
    int inDim;
    if (_layerStack.empty()) {
        inDim = _inputSize;  // antes _inputDimension
    } else {
        inDim = _layerStack.back().getDimension();
    }
    _layerStack.emplace_back(inDim, outDim, false);
}

void MLP::addOutputLayer(int outDim) {
	int inDim;
	if (_layerStack.empty()) {
		inDim = _inputSize;  // antes _inputDimension
	} else {
		inDim = _layerStack.back().getDimension();
	}
	_layerStack.emplace_back(inDim, outDim, true);
}

// =================================================
// MLP: Retropropagación
// =================================================
Vector oneHot(size_t targetLabel, int dimension) {
    std::vector<valueType> encoding(dimension, 0.0);
    if (targetLabel < static_cast<size_t>(dimension)) {
        encoding[targetLabel] = 1.0;
    }
    return Vector(encoding);
}

void MLP::backPropagate(size_t targetLabel) {
    // ----- Step 1: Compute error at the output layer using one-hot encoding -----

    Layer &outputLayer = _layerStack.back();
    // Create the expected (one-hot) vector
    Vector expected = oneHot(targetLabel, outputLayer.size());
    
    // Compute delta at the output layer as: delta = output - expected.
    // Se usa un bucle para modificar directamente el vector de deltas.
    for (int i = 0; i < outputLayer.size(); ++i) {
        outputLayer._deltas[i] = outputLayer._outputs[i] - expected[i];
    }
    
    // ----- Step 2: Backpropagate error through hidden layers -----
    // Iterate from the second-last layer down to the first.
    // Instead of usar bucles anidados de forma clásica, usamos una lambda para procesar cada capa.
    for (int layerIdx = static_cast<int>(_layerStack.size()) - 2; layerIdx >= 0; --layerIdx) {
        Layer &currentLayer = _layerStack[layerIdx];
        Layer &nextLayer    = _layerStack[layerIdx + 1];

        // Para cada neurona de la capa actual, calcular el error acumulado a partir de la siguiente capa.
        for (int i = 0; i < currentLayer.size(); ++i) {
            // Usamos una lambda para acumular el error ponderado de la siguiente capa.
            auto accumulateDelta = [&nextLayer, i](valueType sum, valueType delta, valueType weight) -> valueType {
                return sum + delta * weight;
            };

            // Empleamos un bucle simple para acumular los deltas.
            valueType accumulatedError = 0.0;
            for (int j = 0; j < nextLayer.size(); ++j) {
                accumulatedError += nextLayer._deltas[j] * nextLayer._weights[i][j];
            }
            // Multiplicar el error acumulado por la derivada de la activación de la neurona actual.
            currentLayer._deltas[i] = accumulatedError * currentLayer._valDerivs[i];
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
        for (int i = 0; i < layer._bias.size(); ++i) {
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
		if(!currentLayer.output) {
        currentLayer._outputs   = currentLayer.applyActivation(preActivation);
        currentLayer._valDerivs = currentLayer.applyActivationDeriv(preActivation);
		} else {
		currentLayer._outputs   = currentLayer.applyActivationOutput(preActivation);
		currentLayer._valDerivs = currentLayer.applyActivationDerivOutput(preActivation);
		}
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

    // 1. Extraer el gradiente actual
    valueType grad = layer._grads[row][col];

    // 2. Factores de corrección de sesgo
    valueType b1Correction = 1 - std::pow(layer.beta1, step);
    valueType b2Correction = 1 - std::pow(layer.beta2, step);

    // 3. Actualizar momentos acumulados (m_w_adam, v_w_adam)
    layer._m_w_adam[row][col] = layer.beta1 * layer._m_w_adam[row][col] + (1 - layer.beta1) * grad;
    layer._v_w_adam[row][col] = layer.beta2 * layer._v_w_adam[row][col] + (1 - layer.beta2) * (grad * grad);

    // 4. Corrección de sesgo
    valueType mHat = layer._m_w_adam[row][col] / b1Correction;
    valueType vHat = layer._v_w_adam[row][col] / b2Correction;

    // 5. Actualizar el peso
    layer._weights[row][col] -= _lr * mHat / (std::sqrt(vHat) + layer.eps);
}


// =================================================
// Adam en Bias
// =================================================
void MLP::updateBiasAdam(int idx, int step, Layer &layer) const {
    // Constantes de Adam para el bias
    const valueType beta1 = 0.9;
    const valueType beta2 = 0.999;
    const valueType eps   = 1e-8;

    valueType grad = layer._biasGrads[idx];
    valueType b1Correction = 1 - std::pow(beta1, step);
    valueType b2Correction = 1 - std::pow(beta2, step);

    // Momentos acumulados para bias (m_b_adam, v_b_adam)
    layer._m_b_adam[idx] = beta1 * layer._m_b_adam[idx] + (1 - beta1) * grad;
    layer._v_b_adam[idx] = beta2 * layer._v_b_adam[idx] + (1 - beta2) * (grad * grad);

    valueType mHat = layer._m_b_adam[idx] / b1Correction;
    valueType vHat = layer._v_b_adam[idx] / b2Correction;

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
// MLP: Ajustar la alpha de LeakyReLU
// =================================================
void MLP::setLeakyReLUAlpha(valueType alpha) {
    for (auto &layer : _layerStack) {
        layer._leakyAlpha = alpha;
    }
}
