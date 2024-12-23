#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "matrix.h"

using namespace std;

class Dataset {

    public:

        //CONSTRUCTOR
        Dataset();

        Matrix read_csv(const string& filename);

        Matrix load_labels(const string& filename);

        void write_predictions(const string& filename, const vector<int>& predictions);

        double calculate_accuracy(const Matrix &predictions, const Matrix &labels);

    private:

};
#endif
