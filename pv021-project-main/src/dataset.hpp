#ifndef DATASET_HH
#define DATAST_HH

#include "matrix.hpp"
#include <vector>
#include <string>

class CSVReader {
		
	char _sep;
	vector readRowValues(const std::string &line) const;
    int readRowLabels(const std::string &line) const;

	void normalizeValues(std::vector<vector> &values) const;
    void standardNormalize(std::vector<vector> &values) const;

public:

	CSVReader(char sep = ',') : _sep(sep) {}	
	
	std::vector<vector> readCSVValues(const std::string &filepath);
    std::vector<int> readCSVLabels(const std::string &filepath);
    void exportResults(const std::string &filepath, const std::vector<int> &results);
};

template <typename T>
std::vector<T> readRow(const std::string &line, char sep);

template <typename T = valueType>
T getMinValue(const std::vector<T> &v);

template <typename T = valueType>
T getMaxValue(const std::vector<T> &v);

void displayAccuracy(const std::string &expectedValuesPath, const std::string &actualValuesPath);

#endif
