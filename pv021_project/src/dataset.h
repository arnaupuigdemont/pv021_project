#ifndef DATASET_H
#define DATASET_H

class Dataset {

    public:

        vector<vector<double>> read_csv(const string& filename);

        vector<int> load_labels(const string& filename);

        void write_predictions(const string& filename, const vector<int>& predictions);

};
#endif
