#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include "data_io.h"

// Generate a simple 3-class 2D blob dataset and write to CSV.
int main(int argc, char** argv) {
    int n_per_class = 1500;
    float stddev = 0.7f;
    unsigned seed = 7;
    std::string out = "data/blobs2d_3class.csv";

    if (argc > 1) out = argv[1];

    std::mt19937 rng(seed);
    std::normal_distribution<float> N(0.0f, stddev);

    const int K = 3;
    const int dim = 2;
    const int n = K * n_per_class;
    std::vector<float> X(n * dim);
    std::vector<int> y(n);

    float centers[K][2] = {{-2.0f, 0.0f}, {2.0f, 0.0f}, {0.0f, 2.5f}};
    for (int c = 0; c < K; c++) {
        for (int i = 0; i < n_per_class; i++) {
            int idx = c*n_per_class + i;
            X[idx*2 + 0] = centers[c][0] + N(rng);
            X[idx*2 + 1] = centers[c][1] + N(rng);
            y[idx] = c;
        }
    }

    if (!dataio::write_csv(out, X, y, n, dim)) {
        std::printf("Failed to write %s\n", out.c_str());
        return 1;
    }
    std::printf("Wrote %s (%d rows)\n", out.c_str(), n);
    return 0;
}
