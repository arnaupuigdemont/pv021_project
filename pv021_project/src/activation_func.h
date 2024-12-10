#ifndef ACTIVATION_FUNC_H
#define ACTIVATION_FUNC_H

#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

// activation with relu function 
double relu(double x);

// activation with leaky relu function
double leaky_relu(double x);

double reluDerivative(double x);

void softmax(vector<double> &x);

#endif // ACTIVATION_FUNCTIONS_H