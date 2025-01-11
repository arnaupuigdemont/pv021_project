#ifndef DATASET_HH
#define DATAST_HH

#include "matrix.hpp"
#include <vector>
#include <string>

class dataset {
		
	char _sep;
    int readRowLabels(const std::string &line) const;

	void normalizeValues(std::vector<vector> &values) const;
    void standardNormalize(std::vector<vector> &values) const;

public:

	dataset(char sep = ',') : _sep(sep) {}	
	
	std::vector<vector> readValues(const std::string &filepath);
    std::vector<int> readLabels(const std::string &filepath);
    void writeResults(const std::string &filepath, const std::vector<int> &results);
};
#endif
