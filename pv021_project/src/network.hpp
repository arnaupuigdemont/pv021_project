#ifndef NETWORK_HH
#define NETWORK_HH

#include "layer.hpp"
#include "matrix.hpp"
#include <vector>

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

    std::vector<Vector> _trainData;
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

    void feedForward(const Vector &singleInput);
    void backPropagate(size_t labelIndex);
    void updateWeights(int globalStep);

    void setLeakyReLUAlpha(valueType alpha);

    // Actualizaciones concretas para Adam (privadas):
    void updateWeightAdam(int row, int col, int step, Layer &layer) const;
    void updateBiasAdam(int idx, int step, Layer &layer) const;

	// Actualizaciones concretas para SGD (privadas):
	void updateWeightSGD(int row, int col, int step, Layer &layer) const;
	void updateBiasSGD(int idx, int step, Layer &layer) const;

    // Para recuperar la salida final del MLP en feed-forward
    Vector getMLPOutput();

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
    void train(const std::vector<Vector> &data,
               const std::vector<int> &labels,
               valueType learningRate,
               int epochs,
               int batchSize);

    std::vector<int> predict(const std::vector<Vector> &testData);  
};

// Renombramos "shuffleIndexes" => "shuffleIndices" (opcional)
void shuffleIndices(std::vector<int> &indices);

#endif // NETWORK_HH
