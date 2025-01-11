#ifndef LAYER_HPP
#define LAYER_HPP

#include "matrix.hpp"
#include <vector>
#include <cstddef>

// Renamed 'initializeWeights' to 'initWeights'
Matrix initWeights(int inDim, int outDim, bool output, bool uniformDist = true);

// Renamed 'initializeBias' to 'initBias'
Vector initBias(int dimension);

class Layer {

    friend class MLP; // MLP can access its private members

private:
    bool output;
    int _dimension;  // previously _activationFunction

    Vector _outputs;           // previously _values
    Vector _valDerivs;         // previously _valuesDerivatives

    Matrix _weights;
    Vector _bias;

    Matrix _grads;             // previously _gradients
    Vector _biasGrads;         // previously _biasGradients

    Vector _deltas;

    Matrix _m_w_adam;          // previously _adamFirstMoment
    Matrix _v_w_adam;          // previously _adamSecondMoment
    Vector _m_b_adam;          // previously _adamBiasFirstMom
    Vector _v_b_adam;          // previously _adamBiasSecondMom

    Matrix _sgdVelocity;
    Vector _sgdBiasVelocity;

    valueType _leakyAlpha;     // previously _leakyReLUAlpha

    // Renamed 'useActivationFunction' and 'useDerivedActivationFunction'
    // to something more direct: 'applyActivation' and 'applyActivationDeriv'
    Vector applyActivation(const Vector &inputVec);
    Vector applyActivationOutput(const Vector &inputVec);
    Vector applyActivationDeriv(const Vector &inputVec);
    Vector applyActivationDerivOutput(const Vector &inputVec);

public:
    Layer(int inDim, int outDim, bool output)
      : output(output),
        _dimension(outDim),
        _outputs(outDim),
        _valDerivs(outDim),
        _weights(initWeights(inDim, outDim, output)),
        _bias(initBias(outDim)),
        _grads(inDim, outDim),
        _biasGrads(outDim),
        _deltas(outDim),
        _m_w_adam(inDim, outDim),
        _v_w_adam(inDim, outDim),
        _m_b_adam(outDim),
        _v_b_adam(outDim),
        _sgdVelocity(inDim, outDim),
        _sgdBiasVelocity(outDim),
        _leakyAlpha(0.01) {}
};

#endif // LAYER_HPP