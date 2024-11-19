#ifndef AUX_FUNC_H
#define AUX_FUNC_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;
        
        // Function to read CSV data
        vector<vector<double>> read_csv(const string& filename);

        // Function to load labels from a CSV file
        vector<int> load_labels(const string& filename);

        // Function to write predictions to a CSV file
        void write_predictions(const string& filename, const vector<int>& predictions);

#endif // AUX_FUNC_H