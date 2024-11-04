#include "aux_func.h"

class aux_func {

public:

    //contructor
    aux_func() {}

    // Function to read CSV data
    std::vector<std::vector<double>> read_csv(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        std::vector<std::vector<double>> data;

        while (std::getline(file, line)) {
            std::vector<double> row;
            std::stringstream ss(line);
            std::string value;

            while (std::getline(ss, value, ',')) {
                row.push_back(std::stod(value));
            }
            data.push_back(row);
        }

        return data;
    }

    // Function to load labels from a CSV file
    std::vector<int> load_labels(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        std::vector<int> labels;

        while (std::getline(file, line)) {
            labels.push_back(std::stoi(line));
        }

        return labels;
    }

    // Function to write predictions to a CSV file
    void write_predictions(const std::string& filename, const std::vector<int>& predictions) {
        std::ofstream file(filename);
        for (const int& pred : predictions) {
            file << pred << "\n";
        }
    }
};