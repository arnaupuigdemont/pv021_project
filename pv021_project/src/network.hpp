#ifndef NETWORK_HH
#define NETWORK_HH

#include "matrix.hpp"
#include <vector>

// Renombramos 'activations' a 'ActivationType'
enum class ActivationType { LeakyReLU, Softmax };

// Renombramos 'initialization' a 'WeightInitType'
enum class WeightInitType { He, Glorot };

// Renombramos la función 'getInitializationByActivation' a 'getWeightInitByActivation'
WeightInitType getWeightInitByActivation(ActivationType actFunc);

// Renombramos la función 'initializeWeights' a 'initWeights'
matrix initWeights(int inDim, int outDim, ActivationType actFunc, bool uniformDist = true);

// Renombramos la función 'initializeBias' a 'initBias'
vector initBias(int dimension);

/* 
 * Clase que representa una "Capa" (Layer).
 * Se han renombrado algunas variables:
 *  - _activationFunction -> _actType
 *  - _valuesDerivatives  -> _valDerivs
 */
class Layer {

    friend class MLP; // MLP puede acceder a sus miembros privados

private:
    vector _outputs;           // antes _values
    vector _valDerivs;         // antes _valuesDerivatives

    matrix _weights;
    vector _bias;

    matrix _grads;             // antes _gradients
    vector _biasGrads;         // antes _biasGradients

    vector _deltas;

    matrix _adamFirstMoment;
    matrix _adamSecondMoment;
    vector _adamBiasFirstMom;
    vector _adamBiasSecondMom;

    int    _dimension;
    ActivationType _actType;   // antes _activationFunction

    valueType _leakyAlpha;     // antes _leakyReLUAlpha

    // Renombramos 'useActivationFunction' y 'useDerivedActivationFunction'
    // a algo más directo: 'applyActivation' y 'applyActivationDeriv'
    vector applyActivation(const vector &inputVec);
    vector applyActivationDeriv(const vector &inputVec);

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
        _dimension(outDim),
        _actType(actFunc),
        _leakyAlpha(0.01) {}

    const matrix& getWeights() const { return _weights; }
    const vector& getBias()    const { return _bias; }
    const vector& getOutputs() const { return _outputs; }

    int getDimension() const { return _dimension; }
    size_t size()      const { return static_cast<size_t>(_dimension); }
};

/* 
 * Clase principal "MLP". 
 * - lambda -> regLambda
 * - _inputValues -> _trainData
 * - _inputLabels -> _trainLabels
 * - _layers      -> _layerStack
 * - _inputDimension -> _inputSize
 * - _learningRate   -> _lr
 */
class MLP {

private:
    const double regLambda = 0.1;  // antes lambda

    std::vector<vector> _trainData;
    std::vector<int>    _trainLabels;

    std::vector<Layer>  _layerStack;

    int       _inputSize;   // antes _inputDimension
    valueType _lr;          // antes _learningRate

    // Reordenamos las funciones:
    // 1) feedForward
    // 2) backPropagate
    // 3) updateWeights
    // 4) setLeakyReLUAlpha
    // 5) (Luego las que actualizan pesos con Adam, etc.)

    void feedForward(const vector &singleInput);
    void backPropagate(size_t labelIndex);
    void updateWeights(int globalStep);

    void setLeakyReLUAlpha(valueType alpha);

    // Actualizaciones concretas para Adam (privadas):
    void updateWeightAdam(int row, int col, int step, Layer &layer) const;
    void updateBiasAdam(int idx, int step, Layer &layer) const;

    // Para recuperar la salida final del MLP en feed-forward
    vector getMLPOutput();

public:
    // Constructor
    explicit MLP(int inputDim)
      : regLambda(0.1),
        _inputSize(inputDim),
        _lr(0.01) {}

    // Añadir capa: 
    void addLayer(int outDim, ActivationType actFunc);

    // Obtener capas:
    const std::vector<Layer>& getLayers() { return _layerStack; }

    // Entrenar y predecir
    void train(const std::vector<vector> &data,
               const std::vector<int> &labels,
               valueType learningRate,
               int epochs,
               int batchSize);

    std::vector<int> predict(const std::vector<vector> &testData);  
};

// Renombramos "shuffleIndexes" => "shuffleIndices" (opcional)
void shuffleIndices(std::vector<int> &indices);

#endif // NETWORK_HH
