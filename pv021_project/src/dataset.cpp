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

template <typename T>
T getMaxValue(const std::vector<T> &v) {
	
	T max = -INFINITY;
	
	for (size_t i = 0; i < v.size(); ++i) {
		if (v[i] > max) {
			max = v[i];
		}
	}
	
	return max;
}

template <typename T>
T getMinValue(const std::vector<T> &v) {
	
	T min = INFINITY;
	
	for (size_t i = 0; i < v.size(); ++i) {
		if (v[i] < min) {
			min = v[i];
		}
	}
	
	return min;
}

int dataset::readRowLabels(const std::string &line) const {
	
	std::vector<int> row = readRow<int>(line, _sep);	
    return row.front();
}


std::vector<int> dataset::readCSVLabels(const std::string &filepath) {

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


std::vector<vector> dataset::readCSVValues(const std::string &filepath) {
	
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
		
	for (size_t col = 0; col < values[0].size(); ++col) {
		valueType sum = 0.0;
		valueType mean = 0.0;
		valueType variance = 0.0;
		valueType stddev = 0.0;
		
		for (size_t row = 0; row < values.size(); ++row) {
			sum += values[row][col];
		}
        
		mean = sum / values.size();
		for (size_t row = 0; row < values.size(); ++row) {
			variance += std::pow(values[row][col] - mean, 2);
		}
		variance = variance / values.size();
		stddev = std::sqrt(variance);

		for (size_t row = 0; row < values.size(); ++row) {
			values[row][col] = (values[row][col] - mean) / stddev;
		}
	}
}

void dataset::exportResults(const std::string &filepath, const std::vector<int> &results) {
	std::ofstream os(filepath);
	for (const auto &value : results) {
		os << value << std::endl;
	}
	os.close();
}
