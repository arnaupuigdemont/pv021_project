#include "dataset.hpp"
#include <fstream> 
#include <string>   
#include <iostream>
#include <cmath>

/**
 * @brief read a row from a string
 */
template <typename T>
std::vector<T> readRow(const std::string &line, char sep) {
    std::vector<T> row;
    if (line.empty()) {
        T error = -1;
        return {error};
    }

    size_t index = 0;
    char c;
    std::string currentString;
    T currentValue;

    while (index < line.length()) {
        c = line[index];
        if (c == sep) {
            currentValue = std::stoi(currentString);
            row.emplace_back(currentValue);
            currentString.clear();
        } else {
            currentString += c;
        }
        ++index;
    }

    if (!currentString.empty()) {
        currentValue = std::stoi(currentString);
        row.emplace_back(currentValue);
    }

    return row;
}

/**
 * @brief read labels from a file
 */
std::vector<int> dataset::readLabels(const std::string &filepath) {
    std::vector<int> values;
    std::ifstream is(filepath);
    std::string line;

    if (!is.is_open()) {
        throw std::runtime_error("No se pudo abrir el archivo de labels: " + filepath);
    }

    while (std::getline(is, line)) {
        // Invocar readRow<int>(...)
        auto row = readRow<int>(line, _sep);  
        // Asumimos que la etiqueta está en row.front()
        values.push_back(row.front());
    }

    is.close();
    return values;
}

/**
 * @brief read values from a file
 */
std::vector<Vector> dataset::readValues(const std::string &filepath) 
{
    std::vector<Vector> values;
    std::ifstream is(filepath);
    std::string line;

    if (!is.is_open()) {
        throw std::runtime_error("No se pudo abrir el fichero: " + filepath);
    }

    // Leer cada línea del archivo
    while (std::getline(is, line)) {
        // Leer la fila “in situ” con readRow<valueType>(...),
        // y la emplaces como ‘vector’.
        values.emplace_back(
            readRow<valueType>(line, _sep)  // Devuelve std::vector<valueType>
        );
    }
    is.close();

    // Llamamos a la función que normaliza los datos
    normalizeValues(values);

    return values;
}

/**
 * @brief Normalize values
 */
void dataset::normalizeValues(std::vector<Vector> &values) const {
    if (values.empty()) return; 

    size_t rows = values.size();
    size_t cols = values[0].size();

    for (size_t col = 0; col < cols; ++col) {

        double sum = 0.0;
        double sum_sq = 0.0;
        for (size_t row = 0; row < rows; ++row) {
            double x = values[row][col];
            sum     += x;
            sum_sq  += x * x;
        }

        double mean = sum / rows;
        double variance = (sum_sq / rows) - (mean * mean);
        double stddev   = std::sqrt(variance);

        if (stddev < 1e-12) {
            stddev = 1.0; 
        }

        for (size_t row = 0; row < rows; ++row) {
            values[row][col] = (values[row][col] - mean) / stddev;
        }
    }
}

/**
 * @brief write results to a file
 */
void dataset::writeResults(const std::string &filepath, const std::vector<int> &results) {
	std::ofstream os(filepath);
	for (const auto &value : results) {
		os << value << std::endl;
	}
	os.close();
}
