#ifndef AUX_FUNC_H
#define AUX_FUNC_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>

class aux_func {
    public:

        //contructor
        aux_func() {};
        
        // Function to read CSV data
        std::vector<std::vector<double>> readCsv(const std::string& filename);

        // Function to load labels from a CSV file
        std::vector<int> loadLabels(const std::string& filename);

        // Function to write predictions to a CSV file
        void writePredictions(const std::string& filename, const std::vector<int>& predictions);
};

#endif // AUX_FUNC_H