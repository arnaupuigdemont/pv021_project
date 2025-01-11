#include "dataset.hpp"
#include <fstream> 
#include <string>   
#include <iostream>
#include <cmath>


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
            currentString = "";
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


int dataset::readRowLabels(const std::string &line) const {
	
	std::vector<int> row = readRow<int>(line, _sep);	
    return row.front();
}


std::vector<int> dataset::readLabels(const std::string &filepath) {

    std::vector<int> values;
    std::ifstream is(filepath);
    std::string line;

    while (std::getline(is, line)) {
        values.emplace_back(readRowLabels(line));
    }
    is.close();
    return values;
}

vector dataset::readRowValues(const std::string &line) const {
	
	std::vector<valueType> row = readRow<valueType>(line, _sep);	
	return vector(row);
}	


std::vector<vector> dataset::readValues(const std::string &filepath) {
	
	std::vector<vector> values;
	std::ifstream is(filepath);
	std::string line;
		
	while (std::getline(is, line)) {
		values.emplace_back(readRowValues(line));
	}
	is.close();
	
	normalizeValues(values);
		
	return values;
}

void dataset::normalizeValues(std::vector<vector> &values) const {
    if (values.empty()) return; // Si no hay datos, salir

    size_t rows = values.size();
    size_t cols = values[0].size();

    for (size_t col = 0; col < cols; ++col) {
        // 1. Calcular suma y suma de cuadrados en una sola pasada
        double sum = 0.0;
        double sum_sq = 0.0;
        for (size_t row = 0; row < rows; ++row) {
            double x = values[row][col];
            sum     += x;
            sum_sq  += x * x;
        }

        // 2. Calcular media y desviación estándar
        double mean = sum / rows;
        // var = E(x^2) - (E(x))^2
        double variance = (sum_sq / rows) - (mean * mean);
        double stddev   = std::sqrt(variance);

        // (Opcional) Si stddev es 0 (o muy pequeña), podría causar NaN; manejarlo
        if (stddev < 1e-12) {
            stddev = 1.0; // Evitar división por 0, por ejemplo
        }

        // 3. Normalizar
        for (size_t row = 0; row < rows; ++row) {
            values[row][col] = (values[row][col] - mean) / stddev;
        }
    }
}


void dataset::writeResults(const std::string &filepath, const std::vector<int> &results) {
	std::ofstream os(filepath);
	for (const auto &value : results) {
		os << value << std::endl;
	}
	os.close();
}
