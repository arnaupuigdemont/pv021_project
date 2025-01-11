#ifndef DATASET_HH
#define DATAST_HH

#include "matrix.hpp"
#include <vector>
#include <string>

class dataset {
		;
    int readRowLabels(const std::string &line) const;

	void normalizeValues(std::vector<Vector> &values) const;
    void standardNormalize(std::vector<Vector> &values) const;

public:

	dataset(){}	
	
	std::vector<Vector> readValues(const std::string &filepath);
    std::vector<int> readLabels(const std::string &filepath);
    void writeResults(const std::string &filepath, const std::vector<int> &results);
};
#endif
