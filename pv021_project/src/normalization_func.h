#ifndef NORMALIZATION_FUNC_H
#define NORMALIZATION_FUNC_H

#include <vector>

using namespace std;

// Function to find min and max values in a dataset
pair<vector<double>, vector<double>> find_min_max(const vector<vector<double>> &data);

// Function to normalize a dataset with min-max normalization
void normalize(vector<vector<double>> &data, const vector<double> &minValues, const vector<double> &maxValues);

#endif // NORMALIZATION_FUNC_H