#ifndef LAYER_HH
#define LAYER_HH

#include "matrix.hh"
#include <random>

using namespace std;

class Layer {

    public:

        Matrix weights;
        Matrix biases;

        Layer(int input_size, int output_size);

        Matrix forward_relu(const Matrix &input);

        Matrix forward_leaky_relu(const Matrix &input);

        Matrix forward_softmax(const Matrix &input);

        Matrix backward(const Matrix &grad_output, double learning_rate);

    private:

        Matrix cached_input;

        Matrix relu(const Matrix &input); 

        Matrix leaky_relu(const Matrix &input);

        Matrix softmax(const Matrix &input);

};
#endif