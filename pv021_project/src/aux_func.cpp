#include "aux_func.h"

    // Function to read CSV data
    vector<vector<double>> read_csv(const string& filename) {
        cout << "Reading vectors from " << filename << " ..." << endl;
        ifstream file(filename);
        string line;
        vector<vector<double>> data;
        while (getline(file, line)) {
            stringstream lineStream(line);
            string cell;
            vector<double> row;
            while (getline(lineStream, cell, ',')) {
                row.push_back(stod(cell));
            }
            data.push_back(row);
        }
        if (data.empty()) {
            cerr << filename << " is empty or could not be read." << endl;
        }
        return data;
    }

    // Function to load labels from a CSV file
    vector<int> load_labels(const string& filename) {
        cout << "Reading labels from " << filename << " ..." << endl;
        ifstream file(filename);
        string line;
        vector<int> data;
        while (getline(file, line)) {
            data.push_back(stoi(line));
        }
        if (data.empty()) {
            cerr << filename << " is empty or could not be read." << endl;
        }
        return data;
    }

    // Function to write predictions to a CSV file
    void write_predictions(const string& filename, const vector<int>& predictions) {
        cout << "Writing predictions to " << filename << " ..." << endl;
        ofstream file(filename);
        for (int val : predictions) {
            file << val << endl;
        }
        file.close();
    }