#include "activation_func.h"

using namespace std;

// Returns the maximum of 0 and x
double relu(double x) {
    return std::max(0.0, x);
}

// Returns 1 if x is greater than 0, otherwise returns 0
double reluDerivative(double x) {
    return (x > 0) ? 1.0 : 0.0;
}

// Returns 1 if x is greater than 0, otherwise returns 0
void softmax(vector<double> &x) {
    
    double max_val = *max_element(x.begin(), x.end());
    double sum = 0.0;

    for (auto &val : x) {
        val = exp(val - max_val);
        sum += val;
    }
    for (auto &val : x) {
        val /= sum;
    }
}