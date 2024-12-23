#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include <random>

using namespace std;

class Layer {

    public:

        Matrix weights;
        Matrix biases;

        Layer(int input_size, int output_size);

        Matrix forward(const Matrix &input);

        Matrix backward(const Matrix &grad_output, double learning_rate);

    private:

        Matrix cached_input;

        Matrix relu(const Matrix &input); 

};
#endif