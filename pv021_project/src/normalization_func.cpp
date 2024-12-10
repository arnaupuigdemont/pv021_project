#include "normalization_func.h"

using namespace std;

// Function to find the minimum and maximum values for each feature in the dataset
pair<vector<double>, vector<double>> find_min_max(const vector<vector<double>> &data) {

    cout << "data size: " << data.size() << endl;
    vector<double> minValues = data[0];
    vector<double> maxValues = data[0];

    for (long unsigned int i = 1; i < data.size(); ++i) {

        for (long unsigned int j = 0; j < data[0].size(); ++j) {
            
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

// Function to normalize the dataset using min-max normalization
void normalize(vector<vector<double>> &data, const vector<double> &minValues,  const vector<double> &maxValues) {
    
        for (auto& vector : data) {
        for (auto& val : vector) {
            val /= 255.0;
        }
    }
}