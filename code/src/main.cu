#include <iostream>
#include <cuda_runtime.h>
#include "../include/data_io.h"
#include "../include/thrust_nn.h"

int main(int argc, char *argv[]) {
    // 1. Load data onto device: input matrix X, target vector y
    std::vector<float> X;
    std::vector<int> y;

    int num_entries = 0;
    int num_parameters = 0;

    bool completed = dataio::load_csv_features_labels("../data/blobs2d_3class.csv", X, y, num_entries, num_parameters);

    if (!completed) {
        printf("Error loading data");
        exit(1);
    }

    thrust::device_vector<float> d_X = X;
    thrust::device_vector<int> d_y = y;

    // 2. Initalize hyperparameters + Weights and bias vectors
    float learning_rate = 0.1f;
    int num_epochs = 200;
    int batch_size = 32;
    int num_classes = 3;

    // Host Weights and bias vectors
    std::vector<float> h_W(num_parameters * num_classes);
    std::vector<float> h_b(num_classes);

    // Fill init with random but realistic values
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Standard generator
    std::normal_distribution<float> dist(0.0f, 0.01f);

    for (int i = 0; i < h_W.size(); i++) {
        h_W[i] = dist(gen);
    }

    for (int i = 0; i < h_b.size(); i++) {
        h_b[i] = dist(gen);
    }

    // Device Weights and bias vectors
    thrust::device_vector<float> d_W = h_W;
    thrust::device_vector<float> d_b = h_b;

    // Device logits and probs vectors
    thrust::device_vector<float> d_logits(num_classes, 0.0f);
    thrust::device_vector<float> d_probs(num_classes, 0.0f);

    // Temp row device vectors for logit computation
    thrust::device_vector<float> d_temp_W_row(num_parameters);
    thrust::device_vector<float> d_temp_X_row(num_parameters);

    // 3. Training Loop
    int num_batches = num_entries / batch_size;   
    int batch_start, batch_end, W_row_c;
    float loss, error_c, cost;
    for (int i = 0; i < num_epochs; i++) {
        cost = 0;
        for (int j = 0; j < num_batches; j++) {
            batch_start = j * batch_size;
            batch_end = min(batch_start + batch_size, num_entries);
            for (int s = batch_start; s < batch_end; s++) {

                // Calculate logit and probability of each class
                for (int c = 0; c < num_classes; c++) {
                    W_row_c = c * num_parameters;

                    thrust::copy(d_W.begin() + W_row_c, d_W.begin() + W_row_c + num_parameters, d_temp_W_row.begin());
                    thrust::copy(d_X.begin() + (s * num_parameters), d_X.begin() + (s * num_parameters) + num_parameters, d_temp_X_row.begin());
        
                    d_logits[c] = thrustnn::dot(d_temp_W_row, d_temp_X_row) + d_b[c];
                }
                thrustnn::softmax(d_logits, d_probs);

                // Calculate loss and accumlate into cost
                loss = thrustnn::cross_entropy_one(d_probs, d_y[s]);
                cost += loss;

                // Compute error, update row W and b
                for (int c = 0; c < num_classes; c++) {
                    W_row_c = c * num_parameters;

                    thrust::copy(d_W.begin() + W_row_c, d_W.begin() + W_row_c + num_parameters, d_temp_W_row.begin());
                    error_c = (float)d_probs[c] - (c == d_y[s] ? 1 : 0);
                    thrustnn::saxpy(d_temp_W_row, d_temp_X_row, -learning_rate * error_c);
                    thrust::copy(d_temp_W_row.begin(), d_temp_W_row.end(), d_W.begin() + W_row_c);
                    d_b[c] = d_b[c] - learning_rate * error_c;
                }
            }
        }
        printf("Epoch %d  loss: %.4f\n", i, cost / num_entries);
    }
}