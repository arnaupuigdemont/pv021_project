#include "normalization_func.h"

using namespace std;

pair<vector<double>, vector<double>> find_min_max(const vector<vector<double>> &data) {
    cout << "2" << endl;
    cout << "data size: " << data.size() << endl;
    vector<double> minValues = data[0];
    cout << "3" << endl;
    vector<double> maxValues = data[0];
    cout << "1 " << data.size() << " " << data[0].size() << endl;
    for (long unsigned int i = 1; i < data.size(); ++i) {
        cout << i << endl;
        for (long unsigned int j = 0; j < data[0].size(); ++j) {
            cout << j << endl;
            if (data[i][j] < minValues[j]) {
                minValues[j] = data[i][j];  
            }
            if (data[i][j] > maxValues[j]) {
                maxValues[j] = data[i][j];  
            }
        }
    }
    return {minValues, maxValues};
}

void normalize(vector<vector<double>> &data, const vector<double> &minValues,  const vector<double> &maxValues) {
    for (auto& vector : data) {
        for (size_t i = 0; i < vector.size(); ++i) {
            vector[i] = (vector[i] - minValues[i]) / (maxValues[i] - minValues[i]);
        }
    }
}