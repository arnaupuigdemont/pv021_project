#ifndef LAYER_HPP
#define LAYER_HPP


#include "matrix.hpp"
#include <vector>
#include <cstddef>

// Renombramos la función 'initializeWeights' a 'initWeights'
Matrix initWeights(int inDim, int outDim, bool output, bool uniformDist = true);

// Renombramos la función 'initializeBias' a 'initBias'
Vector initBias(int dimension);

class Layer {

    friend class MLP; // MLP puede acceder a sus miembros privados

private:

    bool output;

    Matrix _outputs;           // antes _values
    Matrix _valDerivs;         // antes _valuesDerivatives

    Matrix _weights;
    Matrix _bias;

    Matrix _grads;             // antes _gradients
    Matrix _biasGrads;         // antes _biasGradients

    Matrix _deltas;

    Matrix _m_w_adam;
    Matrix _v_w_adam;
    Matrix _m_b_adam;
    Matrix _v_b_adam;

    const valueType beta1 = 0.9;
    const valueType beta2 = 0.999;
    const valueType eps   = 1e-8;

	Matrix _sgdVelocity;
	Vector _sgdBiasVelocity;

    int    _dimension;  // antes _activationFunction

    valueType _leakyAlpha;     // antes _leakyReLUAlpha

    // Renombramos 'useActivationFunction' y 'useDerivedActivationFunction'
    // a algo más directo: 'applyActivation' y 'applyActivationDeriv'
    Vector applyActivation(const Vector &inputVec);
    Vector applyActivationOutput(const Vector &inputVec);
    Vector applyActivationDeriv(const Vector &inputVec);
    Vector applyActivationDerivOutput(const Vector &inputVec);

public:
    Layer(int inDim, int outDim, bool output)
      : 
        output(output),
        _outputs(1, outDim),
        _valDerivs(1, outDim),
        _weights(initWeights(inDim, outDim, output)),
        _bias(1, initBias(outDim)),
        _grads(inDim, outDim),
        _biasGrads(1, outDim),
        _deltas(1, outDim),
        _m_w_adam(inDim, outDim),
        _v_w_adam(inDim, outDim),
        _m_b_adam(1, outDim),
        _v_b_adam(1, outDim),
		_sgdVelocity(inDim, outDim),
		_sgdBiasVelocity(outDim),      
        _dimension(outDim),
        _leakyAlpha(0.01) {}

    const Matrix& getWeights() const { return _weights; }
    const Matrix& getBias()    const { return _bias; }
    const Matrix& getOutputs() const { return _outputs; }

    int getDimension() const { return _dimension; }
    size_t size()      const { return static_cast<size_t>(_dimension); }
};
#endif