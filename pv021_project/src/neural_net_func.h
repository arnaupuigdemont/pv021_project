#ifndef NEURAL_NET_FUNC_H
#define NEURAL_NET_FUNC_H

#include <functional>
#include <vector>

using namespace std;

const double LEARNING_RATE = 0.002;
const double BETA1 = 0.9;
const double BETA2 = 0.999;
const double EPSILON = 1e-8;
const double LAMBDA = 1e-4;

void pass_hidden(const vector<vector<double>>& batch_prev, vector<vector<double>>& batch, const vector<double> &weights, const vector<double> &bias, function<double(double)> activation);

void pass_output(const vector<vector<double>>& batch_prev, vector<vector<double>>& batch, const vector<double> &weights, const vector<double> &bias, function<void(vector<double> &x)> activation);

void backpropagation_hidden(const vector<vector<double>>& batch,  vector<vector<double>>& batch_d, const vector<vector<double>>& batch_d_next, const vector<double> &next_layer_weights, function<double(double)> activationDerivative);

void update_weights_Adam(vector<double> &weights, vector<double> &bias, const vector<vector<double>>& batch_gradients, const vector<vector<double>>& batch_inputs, vector<double> &m_weights, vector<double> &v_weights, const int epoch);

void apply_dropout(vector<vector<double>>& batch, double dropout_rate);

#endif // NEURAL_NET_FUNC_H