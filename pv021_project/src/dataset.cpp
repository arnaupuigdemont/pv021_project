#include "dataset.hh"

        //CONSTRUCTOR
        Dataset::Dataset() {}

        Matrix Dataset::read_csv(const std::string &filename) {
            std::cout << "Reading vectors from " << filename << " ..." << std::endl;
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open file " << filename << std::endl;
                return Matrix(0, 0);
            }
            std::string line;
            std::vector<std::vector<double>> data;
            while (std::getline(file, line)) {
                std::stringstream lineStream(line);
                std::string cell;
                std::vector<double> row;
                while (std::getline(lineStream, cell, ',')) {
                    try {
                        row.push_back(std::stod(cell));
                    } catch (const std::invalid_argument &e) {
                        std::cerr << "Error: Invalid data '" << cell << "' in file " << filename << std::endl;
                        return Matrix(0, 0);
                    } catch (const std::out_of_range &e) {
                        std::cerr << "Error: Data out of range '" << cell << "' in file " << filename << std::endl;
                        return Matrix(0, 0);
                    }
                }
                data.push_back(row);
            }
            if (data.empty()) {
                std::cerr << filename << " is empty or could not be read." << std::endl;
                return Matrix(0, 0);
            } else {
                std::cout << "Successfully read " << data.size() << " rows from " << filename << std::endl;
            }
            return Matrix(data);
        }

        Matrix Dataset::read_labels(const std::string &filename) {
            std::cout << "Reading labels from " << filename << " ..." << std::endl;
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open file " << filename << std::endl;
                return Matrix(0, 0);
            }
            std::string line;
            std::vector<std::vector<double>> data;
            while (std::getline(file, line)) {
                try {
                    std::vector<double> row(1, std::stod(line)); // Store each label as a single-row matrix
                    data.push_back(row);
                } catch (const std::invalid_argument &e) {
                    std::cerr << "Error: Invalid data '" << line << "' in file " << filename << std::endl;
                    return Matrix(0, 0);
                } catch (const std::out_of_range &e) {
                    std::cerr << "Error: Data out of range '" << line << "' in file " << filename << std::endl;
                    return Matrix(0, 0);
                }
            }
            if (data.empty()) {
                std::cerr << filename << " is empty or could not be read." << std::endl;
                return Matrix(0, 0);
            } else {
                std::cout << "Successfully read " << data.size() << " labels from " << filename << std::endl;
            }
            return Matrix(data);
        }

        void Dataset::write_predictions(const string& filename, const vector<int>& predictions) {

        
        cout << "Writing predictions to " << filename << " ..." << endl;
        ofstream file(filename);
        for (int val : predictions) {
            file << val << endl;
        }
        file.close();
    }

        double Dataset::calculate_accuracy(const Matrix &predictions, const Matrix &labels) {
            int correct = 0;
            for (int i = 0; i < predictions.getRows(); ++i) {

                int pred_class = std::distance(predictions.data[i].begin(),
                                            std::max_element(predictions.data[i].begin(), predictions.data[i].end()));
                int true_class = std::distance(labels.data[i].begin(),
                                                std::max_element(labels.data[i].begin(), labels.data[i].end()));
                if (pred_class == true_class) correct++;
            }

            return static_cast<double>(correct) / predictions.getRows();
        }

        vector<pair<Matrix, Matrix>> Dataset::create_batches(const Matrix &data, const Matrix &labels, int batch_size) {
            vector<pair<Matrix, Matrix>> batches;
            int total_samples = data.getRows();

            for (int i = 0; i < total_samples; i += batch_size) {
                int end = std::min(i + batch_size, total_samples);
                
                // Extraer las filas correspondientes para el batch
                Matrix batch_data(end - i, data.getCols());
                Matrix batch_labels(end - i, labels.getCols());
                
                for (int j = i; j < end; ++j) {
                    batch_data.data[j - i] = data.data[j];
                    batch_labels.data[j - i] = labels.data[j];
                }
                
                batches.push_back({batch_data, batch_labels});
            }
            return batches;
        }

