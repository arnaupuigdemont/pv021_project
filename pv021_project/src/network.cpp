#include "network.hpp"
#include "matrix.hpp"
#include <vector>
#include <random>  
#include <algorithm>  
#include <cmath>  
#include <iostream>
#include <chrono>


void MLP::train(const std::vector<vector> &inputValues, const std::vector<int> &inputLabels, valueType learningRate, int epochs, int batchSize) {

	_learningRate = learningRate;

	std::vector<vector> batchValues(batchSize);
	std::vector<int> batchLabels(batchSize);	

	int n_examples = inputValues.size();
	int batchCount = n_examples / batchSize;
	
	std::vector<int> indexes(n_examples);
	for (int i = 0; i < n_examples; ++i) {
		indexes[i] = i;
	}

    for (int i = 0; i < epochs; ++i) {

		std::cout << "Epoch " << i + 1 << " / " << epochs << std::endl;

        shuffleIndexes(indexes);        
        
		for (int j = 0; j < batchCount; ++j) {
			
			for (int k = 0; k < batchSize; ++k) {
				int index = indexes[j*batchSize + k];
				batchValues[k] = inputValues[index];
				batchLabels[k] = inputLabels[index];
			}

			_inputValues = batchValues;
			_inputLabels = batchLabels;

			updateWeights(i * batchCount + j + 1);
		}
	}
}

std::vector<int> MLP::predict(const std::vector<vector> &testValues) {

    std::vector<int> predictions;
    vector currentMLPOutput;

    for (const auto& input : testValues) {

        feedForward(input);
        currentMLPOutput = getMLPOutput();
		
        valueType maxValue = -1;
        int currentPrediction = 0;

        for (size_t i = 0; i < currentMLPOutput.size(); ++i) {
			if (currentMLPOutput[i] >= maxValue) {
                maxValue = currentMLPOutput[i];
                currentPrediction = i;
			}
		}
		predictions.emplace_back(currentPrediction);
    }

    return predictions;
}


void shuffleIndexes(std::vector<int> &indexes) {

	std::random_device rd;
	std::mt19937 generator(0);
	std::shuffle(indexes.begin(), indexes.end(), generator);
}



vector MLP::getMLPOutput() {
    return _layers.back().getValues();
}



void MLP::addLayer(int dimension, activations activationFunction) {

    int oldDimension;
	if (_layers.empty()) {
		oldDimension = _inputDimension;
	} else {
		oldDimension = _layers.back().dimension();
	}

	_layers.emplace_back(oldDimension, dimension, activationFunction);
}


void MLP::backPropagate(size_t inputLabel) {

    for (size_t i = 0; i < _layers.back().size(); ++i) {
		
		valueType inputLabelHotEncoded;
		if (inputLabel == i) {
			inputLabelHotEncoded = 1.0;
		} else {
			inputLabelHotEncoded = 0.0;
		}

		_layers.back()._deltas[i] = _layers.back()._values[i] - inputLabelHotEncoded;		
	}
				
	for (int l = (int)_layers.size() - 2; l >= 0; --l) {
		for (size_t i = 0; i < _layers[l].size(); ++i) {  
			valueType deltasSum = 0;
			for (size_t j = 0; j < _layers[l+1].size(); ++j) {											
				deltasSum += _layers[l+1]._deltas[j] * _layers[l+1]._weights[i][j];
			}
			_layers[l]._deltas[i] = deltasSum * _layers[l]._valuesDerivatives[i];
		}
	}
}


void MLP::updateWeights(int step) {

	for (size_t k = 0; k < _inputValues.size(); ++k) {
		
		auto inputValue = _inputValues[k];
		auto inputLabel = _inputLabels[k];
		feedForward(inputValue);
		backPropagate(inputLabel);

		for (size_t l = 0; l < _layers.size(); ++l) {

			auto layerBelowValues = (l == 0) ? inputValue : _layers[l - 1]._values;
			
            for (int i = 0; i < _layers[l]._weights.rows(); ++i) {				
				for (int j = 0; j < _layers[l]._weights.cols(); ++j) {							
                    _layers[l]._gradients[i][j] += _layers[l]._deltas[j] * layerBelowValues[i];					
				}
			}
			
			for (size_t j = 0; j < _layers[l].size(); ++j) {
				_layers[l]._biasGradients[j] += _layers[l]._deltas[j]; 
			}
		}
	}

	for (auto &layer : _layers) {
		
		#pragma omp parallel for num_threads(16)
		for (int i = 0; i < layer._weights.rows(); ++i) {
			for (int j = 0; j < layer._weights.cols(); ++j) {
				updateWeightAdam(i, j, step, layer);
				layer._gradients[i][j] = 0;
			}
		}

		for (size_t i = 0; i < layer._bias.size(); ++i) {
			updateBiasAdam(i, step, layer);
			layer._biasGradients[i] = 0;
		}
	}
}


void MLP::feedForward(const vector &input) {

	vector innerPotential;
	for (size_t i = 0; i < _layers.size(); ++i) {

		if (i == 0) {
			innerPotential = (input * _layers[i]._weights) + _layers[i]._bias;
		} else {
			innerPotential = (_layers[i-1]._values * _layers[i]._weights) + _layers[i]._bias;
		}
		
		_layers[i]._values = _layers[i].useActivationFunction(innerPotential);
		_layers[i]._valuesDerivatives = _layers[i].useDerivedActivationFunction(innerPotential);		
	}	
}


initialization getInitializationByActivation(activations activationFunction) {

	switch (activationFunction) {
        case activations::_leakyReLu:
			return initialization::he;

		case activations::_softmax:
        default:
			return initialization::glorot;
	}
}


matrix initializeWeights(int n, int m, activations activationFunction, bool uniformDistribution) {

	matrix weights(n, m);

	initialization init = getInitializationByActivation(activationFunction);
	valueType multiplier = (uniformDistribution) ? 3.0 : 1.0;
	valueType upperBound;

	switch (init) {
		case initialization::glorot:
			upperBound = std::sqrt(multiplier * 2.0 / (n + m));
            break;
        case initialization::he:
            upperBound = std::sqrt(multiplier * 2.0 / n);
            break;
    }
	valueType lowerBound = -upperBound;

	std::random_device rd;
	std::mt19937 generator(0);
	std::uniform_real_distribution<valueType> distribution(lowerBound, upperBound);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			weights[i][j] = distribution(generator);
		}
	}

	return weights;
}


vector initializeBias(int dimension) {

    vector bias(dimension); 
    return bias;
}


void MLP::updateWeightAdam(int i, int j, int step, Layer& layer) const {

	valueType beta_1 = 0.9;
	valueType beta_2 = 0.999;
	valueType epsilon = 1e-8;  

	valueType beta_1_t = std::pow(beta_1, step);
	valueType beta_2_t = std::pow(beta_2, step);

	valueType gradient = layer._gradients[i][j];

	layer._adamFirstMoment[i][j] = (beta_1) * layer._adamFirstMoment[i][j] + (1 - beta_1) * gradient;
	layer._adamSecondMoment[i][j] = (beta_2) * layer._adamSecondMoment[i][j] + (1 - beta_2) * (gradient * gradient);

	auto biasCorrectedPastGradient = layer._adamFirstMoment[i][j] / (1 - beta_1_t);
	auto biasCorrectedPastSquaredGradient = layer._adamSecondMoment[i][j] / (1 - beta_2_t);

    layer._weights[i][j] -= _learningRate / (std::sqrt(biasCorrectedPastSquaredGradient) + epsilon) * biasCorrectedPastGradient;
}
	
void MLP::updateBiasAdam(int i, int step, Layer& layer) const {

	valueType beta_1 = 0.9;
	valueType beta_2 = 0.999;
	valueType epsilon = 1e-8;
	valueType beta_1_t = std::pow(beta_1, step);
	valueType beta_2_t = std::pow(beta_2, step);
	
	valueType gradient = layer._biasGradients[i];
	
	layer._adamBiasFirstMoment[i] = (beta_1) * layer._adamBiasFirstMoment[i] + (1 - beta_1) * gradient;
    layer._adamBiasSecondMoment[i] = (beta_2) * layer._adamBiasSecondMoment[i] + (1 - beta_2) * (gradient * gradient);

    auto biasCorrectedPastGradient = layer._adamBiasFirstMoment[i] / (1 - beta_1_t);
    auto biasCorrectedPastSquaredGradient = layer._adamBiasSecondMoment[i] / (1 - beta_2_t);

    layer._bias[i] -= _learningRate / (std::sqrt(biasCorrectedPastSquaredGradient) + epsilon) * biasCorrectedPastGradient;
}

vector Layer::useActivationFunction(const vector &vec) {

    switch (_activationFunction) {

	    case activations::_leakyReLu:
            return leakyReLu(vec, _leakyReLUAlpha);
		case activations::_softmax:		    
			return softmax(vec);
		default:
			return vector(vec.size());
	}
}

vector Layer::useDerivedActivationFunction(const vector &vec) {

    switch (_activationFunction) {
        
        case activations::_leakyReLu:
            return leakyReLuDerivative(vec, _leakyReLUAlpha);
        case activations::_softmax:
            return softmaxDerivative(_values);
        default:
            return vector(vec.size());
    }
}


void MLP::setLeakyReLUAlpha(valueType alpha) {

    for (auto &layer : _layers) {
        layer._leakyReLUAlpha = alpha;
    }
}


