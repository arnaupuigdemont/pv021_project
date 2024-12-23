#ifndef DATASET_HH
#define DATASET_HH

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "matrix.hh"

using namespace std;

class Dataset {

    public:

        //CONSTRUCTOR
        Dataset();

        Matrix read_csv(const string& filename);

        Matrix read_labels(const string& filename);

        void write_predictions(const string& filename, const vector<int>& predictions);

        double calculate_accuracy(const Matrix &predictions, const Matrix &labels);

        vector<pair<Matrix, Matrix>> create_batches(const Matrix &data, const Matrix &labels, int batch_size);

    private:

};
#endif
