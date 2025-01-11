#ifndef NETWORK_HH
#define NETWORK_HH

#include "matrix.hpp"
#include <vector>

enum activations {_leakyReLu, _softmax};
enum initialization {he, glorot};

initialization getInitializationByActivation(activations activationFunction);
matrix initializeWeights(int n, int m, activations activationFunction, bool uniformDistribution = true);
vector initializeBias(int dimension);

class Layer {

	friend class MLP; 
	
	vector _values;
	vector _valuesDerivatives;
	
	matrix _weights;
	vector _bias;
	matrix _gradients;
	vector _biasGradients;
	
	vector _deltas;
	
	matrix _adamFirstMoment;
	matrix _adamSecondMoment;
	vector _adamBiasFirstMoment;
	vector _adamBiasSecondMoment;
	
	int _dimension;
	activations _activationFunction;

	valueType _leakyReLUAlpha;
	
	vector useActivationFunction(const vector &vec);
	vector useDerivedActivationFunction(const vector &vec);

public:
	
	Layer(int oldDimension, int dimension, activations activationFunction)
		: _values(dimension),
          _valuesDerivatives(dimension),
          _weights(initializeWeights(oldDimension, dimension, activationFunction)),
          _bias(initializeBias(dimension)),
          _gradients(oldDimension, dimension),
          _biasGradients(dimension),
          _deltas(dimension),
          _adamFirstMoment(oldDimension, dimension),
          _adamSecondMoment(oldDimension, dimension),
          _adamBiasFirstMoment(dimension),
          _adamBiasSecondMoment(dimension),
          _dimension(dimension),
          _activationFunction(activationFunction),
          _leakyReLUAlpha(0.01) {}

	const matrix& getWeights() const { return _weights; }
	const vector& getBias() const { return _bias; }
	const vector& getValues() const { return _values; }			
	int dimension() const { return _dimension; }
	size_t size() const { return _dimension; }
};

class MLP {

	const double lambda = 0.001;

	std::vector<vector> _inputValues;
	std::vector<int> _inputLabels;
	std::vector<Layer> _layers;
	int _inputDimension;
	valueType _learningRate;
	
	void feedForward(const vector &input);
	void backPropagate(size_t label);
	void updateWeights(int step);		
	
	void updateWeightAdam(int i, int j, int step, Layer& layer) const;
	void updateBiasAdam(int i, int step, Layer& layer) const;
	
	vector getMLPOutput();

	void setLeakyReLUAlpha(valueType alpha);

public:
	
	explicit MLP (int inputDimension)
	    : _inputDimension(inputDimension),
	      _learningRate(0.01) {}
	
	void addLayer(int dimension, activations activationFunction);
	const std::vector<Layer>& getLayers() { return _layers; }
	
	void train(const std::vector<vector> &inputValues, const std::vector<int> &inputLabels, valueType learningRate, int epochs, int batchSize);	
	std::vector<int> predict(const std::vector<vector> &testValues);			
};

void shuffleIndexes(std::vector<int> &indexes);

#endif
