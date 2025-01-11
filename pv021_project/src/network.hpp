#ifndef NETWORK_HH
#define NETWORK_HH

#include "layer.hpp"
#include "matrix.hpp"
#include <vector>


class MLP {

private:
    const double regLambda = 0.5;  

    std::vector<Vector> _trainData;
    std::vector<int>    _trainLabels;

    std::vector<Layer> _layerStack; // Stack of layers

    int _inputSize;   // Input size
    valueType _lr;    // Learning rate

    void feedForward(const Vector &singleInput); // Forward pass
    void backPropagate(size_t labelIndex); // Backward pass
    void updateWeights(int globalStep); // Update weights

    void setLeakyReLUAlpha(valueType alpha); // Set alpha for LeakyReLU

    // Adam weight update
    void updateWeightAdam(int row, int col, int step, Layer &layer) const;
    // Adam bias update
    void updateBiasAdam(int idx, int step, Layer &layer) const;

    // SGD weight update
    void updateWeightSGD(int row, int col, int step, Layer &layer) const;
    // SGD bias update
    void updateBiasSGD(int idx, int step, Layer &layer) const;

    // Get final MLP output
    Vector getMLPOutput();

public:
    // Constructor
    explicit MLP(int inputDim)
      : regLambda(0.1),
        _inputSize(inputDim),
        _lr(0.01) {}

    // Add layer
    void addLayer(int outDim);
	void addOutputLayer(int outDim);

    // Train the network
    void train(const std::vector<Vector> &trainData,
               const std::vector<int> &trainLabels,
               valueType lr,
               int epochs,
               int batchSize);

    // Predict labels for test data
    std::vector<int> predict(const std::vector<Vector> &testData);  
};

// Renombramos "shuffleIndexes" => "shuffleIndices" (opcional)
void shuffleIndices(std::vector<int> &indices);

#endif // NETWORK_HH
