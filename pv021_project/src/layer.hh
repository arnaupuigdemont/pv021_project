#ifndef LAYER_HH
#define LAYER_HH

#include "matrix.hh"
#include <random>

using namespace std;

class Layer {

    public:

        Matrix weights;
        Matrix biases;

        Matrix m_weights;
        Matrix m_biases;
        Matrix v_weights;  
        Matrix v_biases;

        Matrix momentum_weights;
        Matrix momentum_biases;

        double beta1 = 0.9;
        double beta2 = 0.999;
        double epsilon = 1e-8;
        int t = 0;

        Layer(int input_size, int output_size);

        Matrix forward_relu(const Matrix &input);
        Matrix forward_leaky_relu(const Matrix &input);
        Matrix forward_softmax(const Matrix &input);

        double compute_l2_penalty() const;

        Matrix backward(const Matrix &grad_output, double learning_rate);
        Matrix backward_ADAM(const Matrix &grad_output, double learning_rate, double lambda);
        Matrix backward_ADAM_hidden(const Matrix &grad_output, double learning_rate, double lambda);
        Matrix backward_ADAM_output(const Matrix &output, const Matrix &target, double learning_rate, double lambda);

        Matrix getWeights();
        Matrix getBiases();

    private:

        Matrix cached_input;

        Matrix relu(const Matrix &input); 
        Matrix leaky_relu(const Matrix &input);
        Matrix leaky_relu_derivative(const Matrix &input);
        Matrix softmax(const Matrix &input);
        Matrix softmax_derivative(const Matrix &input);

};
#endif