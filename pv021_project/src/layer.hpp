#ifndef LAYER_HPP
#define LAYER_HPP


#include "matrix.hpp"
#include <vector>
#include <cstddef>

// Renombramos 'activations' a 'ActivationType'
enum class ActivationType { LeakyReLU, Softmax };

// Renombramos 'initialization' a 'WeightInitType'
enum class WeightInitType { He, Glorot };

// Renombramos la función 'getInitializationByActivation' a 'getWeightInitByActivation'
WeightInitType getWeightInitByActivation(ActivationType actFunc);

// Renombramos la función 'initializeWeights' a 'initWeights'
Matrix initWeights(int inDim, int outDim, ActivationType actFunc, bool uniformDist = true);

// Renombramos la función 'initializeBias' a 'initBias'
Vector initBias(int dimension);

class Layer {

    friend class MLP; // MLP puede acceder a sus miembros privados

private:
    Vector _outputs;           // antes _values
    Vector _valDerivs;         // antes _valuesDerivatives

    Matrix _weights;
    Vector _bias;

    Matrix _grads;             // antes _gradients
    Vector _biasGrads;         // antes _biasGradients

    Vector _deltas;

    Matrix _adamFirstMoment;
    Matrix _adamSecondMoment;
    Vector _adamBiasFirstMom;
    Vector _adamBiasSecondMom;

	Matrix _sgdVelocity;
	Vector _sgdBiasVelocity;

    int    _dimension;
    ActivationType _actType;   // antes _activationFunction

    valueType _leakyAlpha;     // antes _leakyReLUAlpha

    // Renombramos 'useActivationFunction' y 'useDerivedActivationFunction'
    // a algo más directo: 'applyActivation' y 'applyActivationDeriv'
    Vector applyActivation(const Vector &inputVec);
    Vector applyActivationDeriv(const Vector &inputVec);

public:
    Layer(int inDim, int outDim, ActivationType actFunc)
      : _outputs(outDim),
        _valDerivs(outDim),
        _weights(initWeights(inDim, outDim, actFunc)),
        _bias(initBias(outDim)),
        _grads(inDim, outDim),
        _biasGrads(outDim),
        _deltas(outDim),
        _adamFirstMoment(inDim, outDim),
        _adamSecondMoment(inDim, outDim),
        _adamBiasFirstMom(outDim),
        _adamBiasSecondMom(outDim),
		_sgdVelocity(inDim, outDim),
		_sgdBiasVelocity(outDim),
        _dimension(outDim),
        _actType(actFunc),
        _leakyAlpha(0.01) {}

    const Matrix& getWeights() const { return _weights; }
    const Vector& getBias()    const { return _bias; }
    const Vector& getOutputs() const { return _outputs; }

    int getDimension() const { return _dimension; }
    size_t size()      const { return static_cast<size_t>(_dimension); }
};
#endif