#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>

// Very small utilities for teaching. Not production-grade parsing.
namespace dataio {

// Load a simple CSV with floats and last column as integer label.
// Returns: X flattened row-major (n*dim), y labels (n), and sets n, dim.
inline bool load_csv_features_labels(const std::string& path,
                                    std::vector<float>& X,
                                    std::vector<int>& y,
                                    int& n, int& dim) {
    std::ifstream in(path);
    if (!in) return false;

    std::string line;
    std::vector<std::vector<float>> rows;
    std::vector<int> labels;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> vals;
        while (std::getline(ss, cell, ',')) {
            vals.push_back(std::stof(cell));
        }
        if (vals.size() < 2) continue;
        int lbl = static_cast<int>(vals.back());
        vals.pop_back();
        labels.push_back(lbl);
        rows.push_back(std::move(vals));
    }

    n = static_cast<int>(rows.size());
    if (n == 0) return false;
    dim = static_cast<int>(rows[0].size());

    X.resize(n * dim);
    y.resize(n);

    for (int i = 0; i < n; i++) {
        if ((int)rows[i].size() != dim) return false;
        for (int j = 0; j < dim; j++) {
            X[i * dim + j] = rows[i][j];
        }
        y[i] = labels[i];
    }
    return true;
}

// Generate a 2D synthetic dataset (two Gaussians / blobs) for binary classification.
// label 0 centered at (-1,0), label 1 centered at (1,0).
inline void make_blobs_binary(int n_per_class, float stddev, unsigned seed,
                              std::vector<float>& X, std::vector<int>& y) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.0f, stddev);

    int n = 2 * n_per_class;
    int dim = 2;
    X.resize(n * dim);
    y.resize(n);

    for (int i = 0; i < n_per_class; i++) {
        float x0 = -1.0f + N(rng);
        float x1 =  0.0f + N(rng);
        X[i*2+0] = x0; X[i*2+1] = x1; y[i] = 0;
    }
    for (int i = 0; i < n_per_class; i++) {
        float x0 =  1.0f + N(rng);
        float x1 =  0.0f + N(rng);
        int idx = n_per_class + i;
        X[idx*2+0] = x0; X[idx*2+1] = x1; y[idx] = 1;
    }
}

// Write dataset to CSV (features...,label)
inline bool write_csv(const std::string& path,
                      const std::vector<float>& X,
                      const std::vector<int>& y,
                      int n, int dim) {
    std::ofstream out(path);
    if (!out) return false;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < dim; j++) {
            out << X[i*dim + j] << ",";
        }
        out << y[i] << "\n";
    }
    return true;
}

} // namespace dataio
