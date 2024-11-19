#include "aux_func.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

// Function to read CSV data
vector<vector<double>> read_csv(const string& filename) {
    cout << "Reading vectors from " << filename << " ..." << endl;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return {};
    }

    string line;
    vector<vector<double>> data;
    while (getline(file, line)) {
        stringstream lineStream(line);
        string cell;
        vector<double> row;
        while (getline(lineStream, cell, ',')) {
            try {
                row.push_back(stod(cell));
            } catch (const invalid_argument& e) {
                cerr << "Error: Invalid data '" << cell << "' in file " << filename << endl;
                return {};
            } catch (const out_of_range& e) {
                cerr << "Error: Data out of range '" << cell << "' in file " << filename << endl;
                return {};
            }
        }
        data.push_back(row);
    }

    if (data.empty()) {
        cerr << filename << " is empty or could not be read." << endl;
    } else {
        cout << "Successfully read " << data.size() << " rows from " << filename << endl;
    }

    return data;
}

// Function to load labels from a CSV file
vector<int> load_labels(const string& filename) {
    cout << "Reading labels from " << filename << " ..." << endl;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return {};
    }

    string line;
    vector<int> data;
    while (getline(file, line)) {
        try {
            data.push_back(stoi(line));
        } catch (const invalid_argument& e) {
            cerr << "Error: Invalid data '" << line << "' in file " << filename << endl;
            return {};
        } catch (const out_of_range& e) {
            cerr << "Error: Data out of range '" << line << "' in file " << filename << endl;
            return {};
        }
    }

    if (data.empty()) {
        cerr << filename << " is empty or could not be read." << endl;
    } else {
        cout << "Successfully read " << data.size() << " labels from " << filename << endl;
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