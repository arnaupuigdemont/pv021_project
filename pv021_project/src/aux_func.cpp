#include "aux_func.h"

    // Function to read CSV data
    vector<vector<double>> read_csv(const string& filename) {
        ifstream file(filename);
        string line;
        vector<vector<double>> data;

        while (getline(file, line)) {
            vector<double> row;
            stringstream ss(line);
            string value;

            while (getline(ss, value, ',')) {
                row.push_back(stod(value));
            }
            data.push_back(row);
        }

        return data;
    }

    // Function to load labels from a CSV file
    vector<int> load_labels(const string& filename) {
        ifstream file(filename);
        string line;
        vector<int> labels;

        while (getline(file, line)) {
            labels.push_back(stoi(line));
        }

        return labels;
    }

    // Function to write predictions to a CSV file
    void write_predictions(const string& filename, const vector<int>& predictions) {
        ofstream file(filename);
        for (const int& pred : predictions) {
            file << pred << "\n";
        }
    }