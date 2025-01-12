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

    friend class Network; // MLP puede acceder a sus miembros privados

private:

    bool output;

    int _size;

    Vector _outputs;          
    Vector _valDerivs;        

    Matrix _weights;
    Vector _bias;

    Matrix _grads;             
    Vector _biasGrads;         

    Vector _deltas;

    Matrix _m_w_adam;
    Matrix _v_w_adam;
    Vector _m_b_adam;
    Vector _v_b_adam;

    const valueType beta1 = 0.9;
    const valueType beta2 = 0.999;
    const valueType eps   = 1e-8;

	Matrix _sgdVelocity;
	Vector _sgdBiasVelocity; 

    valueType _leakyAlpha;     

    Vector applyActivation(const Vector &inputVec);
    Vector applyActivationOutput(const Vector &inputVec);
    Vector applyActivationDeriv(const Vector &inputVec);
    Vector applyActivationDerivOutput(const Vector &inputVec);

public:
    Layer(int inDim, int outDim, bool output)
      : 
        output(output),
        _size(outDim),
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

    const Matrix& getWeights() const { return _weights; }
    const Vector& getBias()    const { return _bias; }
    const Vector& getOutputs() const { return _outputs; }

    int getSize() const { return _size; }
    size_t size()      const { return static_cast<size_t>(_size); }
};
#endif