#include <iostream>
#include <random>
#include <thrust/extrema.h>
#include <cuda_runtime.h>
#include "../include/data_io.h"
#include "../include/thrust_nn.h"
#include <cstdlib>
#include <string>
#include <algorithm>

int main(int argc, char *argv[]) {
    // MILESTONE 1: Stabilized Softmax Classifier
    // MILESTONE 3: Adding Cuda Event Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <dataset.csv> <num_epochs>\n";
        return 1;
    }

    // a. Load data onto device: input matrix X, target vector y

    std::vector<float> X;
    std::vector<int> y;

    int num_entries = 0;
    int num_parameters = 0;

    bool completed = dataio::load_csv_features_labels(argv[1], X, y, num_entries, num_parameters);

    if (!completed) {
        printf("Error loading data\n");
        exit(1);
    }

    cudaEventRecord(start);
    thrust::device_vector<float> d_X = X;
    thrust::device_vector<int> d_y = y;

    // b. Initalize hyperparameters + Weights and bias vectors
    float learning_rate = 0.1f;

    int num_epochs = atoi(argv[2]);
    if (num_epochs <= 0 || num_epochs > 10000) {
        std::cerr << "Epoch count must be between 1 and 10000\n";
        return 1;
    }

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

    // c. Training Loop
    int num_batches = num_entries / batch_size;   
    int batch_start, batch_end, W_row_c, W_row_h;
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

    // d. Evaluation
    int correct = 0;
    for (int s = 0; s < num_entries; s++) {
        for (int c = 0; c < num_classes; c++) {
            W_row_c = c * num_parameters;

            thrust::copy(d_W.begin() + W_row_c, d_W.begin() + W_row_c + num_parameters, d_temp_W_row.begin());
            thrust::copy(d_X.begin() + (s * num_parameters), d_X.begin() + (s * num_parameters) + num_parameters, d_temp_X_row.begin());

            d_logits[c] = thrustnn::dot(d_temp_W_row, d_temp_X_row) + d_b[c];
        }
        thrustnn::softmax(d_logits, d_probs);

        auto max_it = thrust::max_element(d_probs.begin(), d_probs.end());
        int predicted = max_it - d_probs.begin();
        if (predicted == d_y[s]) correct++;
    }
    float soft_max_acc = 100.0f * correct / num_entries;
    printf("Softmax Accuracy: %.2f%%\n", soft_max_acc);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    printf("Dataset: %s\n", argv[1]);
    printf("Softmax Classifier GPU time: %.2f seconds\n", ms / 1000.0f);

    // MILESTONE 2: 1 Hidden Layer MLP
    // a. Init W1, W2, b1, b2, temp hidden vector
    int hidden_layer_size = 8;
    learning_rate = .01f;

    // Host Weights and bias vectors
    std::vector<float> h_W1(num_parameters * hidden_layer_size);
    std::vector<float> h_b1(hidden_layer_size);
    std::vector<float> h_W2(hidden_layer_size * num_classes);
    std::vector<float> h_b2(num_classes);

    // Fill init with random but realistic values
    for (int i = 0; i < h_W1.size(); i++) {
        h_W1[i] = dist(gen);
    }
    for (int i = 0; i < h_W2.size(); i++) {
        h_W2[i] = dist(gen);
    }

    for (int i = 0; i < h_b1.size(); i++) {
        h_b1[i] = dist(gen);
    }
    for (int i = 0; i < h_b2.size(); i++) {
        h_b2[i] = dist(gen);
    }

    // Device Weights and bias vectors
    cudaEventRecord(start);
    thrust::device_vector<float> d_W1 = h_W1;
    thrust::device_vector<float> d_W2 = h_W2;
    thrust::device_vector<float> d_b1 = h_b1;
    thrust::device_vector<float> d_b2 = h_b2;
    thrust::device_vector<float> d_hidden(hidden_layer_size, 0.0f);
    thrust::device_vector<float> d_hidden_vals(hidden_layer_size, 0.0f);
    thrust::device_vector<float> d_error(num_classes, 0.0f);

    thrust::device_vector<float> d_temp_W1_row(num_parameters);
    thrust::device_vector<float> d_temp_X1_row(num_parameters);
    thrust::device_vector<float> d_temp_W2_row(hidden_layer_size);
    thrust::device_vector<float> d_temp_X2_row(hidden_layer_size);

    // b. Training Loop
    for (int i = 0; i < num_epochs; i++) {
        cost = 0;
        for (int j = 0; j < num_batches; j++) {
            batch_start = j * batch_size;
            batch_end = min(batch_start + batch_size, num_entries);
            for (int s = batch_start; s < batch_end; s++) {

                // Forward Propogation
                // Hidden vector = W1*X + b1
                for (int h = 0; h < hidden_layer_size; h++) {
                    W_row_h = h * num_parameters;

                    thrust::copy(d_W1.begin() + W_row_h, d_W1.begin() + W_row_h + num_parameters, d_temp_W1_row.begin());
                    thrust::copy(d_X.begin() + (s * num_parameters), d_X.begin() + (s * num_parameters) + num_parameters, d_temp_X1_row.begin());
        
                    d_hidden[h] = thrustnn::dot(d_temp_W1_row, d_temp_X1_row) + d_b1[h];
                    d_hidden_vals[h] = d_hidden[h];
                }

                // ReLU
                thrustnn::relu_inplace(d_hidden);

                // Output vector = W2*HIDDEN + b2
                for (int c = 0; c < num_classes; c++) {
                    W_row_c = c * hidden_layer_size;

                    thrust::copy(d_W2.begin() + W_row_c, d_W2.begin() + W_row_c + hidden_layer_size, d_temp_W2_row.begin());
                    thrust::copy(d_hidden.begin(), d_hidden.begin() + hidden_layer_size, d_temp_X2_row.begin());
        
                    d_logits[c] = thrustnn::dot(d_temp_W2_row, d_temp_X2_row) + d_b2[c];
                }

                // Softmax
                thrustnn::softmax(d_logits, d_probs);

                loss = thrustnn::cross_entropy_one(d_probs, d_y[s]);
                cost += loss;

                // Backpropogation (W/b updating)
                // Compute output layer error, update row W2 and b2
                for (int c = 0; c < num_classes; c++) {
                    W_row_c = c * hidden_layer_size;

                    thrust::copy(d_W2.begin() + W_row_c, d_W2.begin() + W_row_c + hidden_layer_size, d_temp_W2_row.begin());
                    thrust::copy(d_hidden.begin(), d_hidden.end(), d_temp_X2_row.begin());
                    error_c = (float)d_probs[c] - (c == d_y[s] ? 1 : 0);
                    d_error[c] = error_c;
                    thrustnn::saxpy(d_temp_W2_row, d_temp_X2_row, -learning_rate * error_c);
                    thrust::copy(d_temp_W2_row.begin(), d_temp_W2_row.end(), d_W2.begin() + W_row_c);
                    d_b2[c] = d_b2[c] - learning_rate * error_c;
                }

                // Update update row W1 and b1
                float grad_h;
                for (int h = 0; h < hidden_layer_size; h++) {
                    grad_h = 0;
                    for (int c = 0; c < num_classes; c++) {
                        grad_h += d_error[c] * (float)d_W2[hidden_layer_size * c + h];     
                    }
                    if ((float)d_hidden_vals[h] <= 0) {
                        grad_h = 0;
                    }

                    W_row_h = h * num_parameters;

                    thrust::copy(d_W1.begin() + W_row_h, d_W1.begin() + W_row_h + num_parameters, d_temp_W1_row.begin());
                    thrust::copy(d_X.begin() + (s * num_parameters), d_X.begin() + (s * num_parameters) + num_parameters, d_temp_X1_row.begin());
                    thrustnn::saxpy(d_temp_W1_row, d_temp_X1_row, -learning_rate * grad_h);
                    thrust::copy(d_temp_W1_row.begin(), d_temp_W1_row.end(), d_W1.begin() + W_row_h);
                    d_b1[h] = d_b1[h] - learning_rate * grad_h;
                }
                
            }
        }
        printf("MLP Epoch %d  loss: %.4f\n", i, cost / num_entries);
    }

    // c. Evaluation
    correct = 0;
    for (int s = 0; s < num_entries; s++) {
        for (int h = 0; h < hidden_layer_size; h++) {
            W_row_h = h * num_parameters;

            thrust::copy(d_W1.begin() + W_row_h, d_W1.begin() + W_row_h + num_parameters, d_temp_W1_row.begin());
            thrust::copy(d_X.begin() + (s * num_parameters), d_X.begin() + (s * num_parameters) + num_parameters, d_temp_X1_row.begin());

            d_hidden[h] = thrustnn::dot(d_temp_W1_row, d_temp_X1_row) + d_b1[h];
        }

        // ReLU
        thrustnn::relu_inplace(d_hidden);

        // Output vector = W2*HIDDEN + b2
        for (int c = 0; c < num_classes; c++) {
            W_row_c = c * hidden_layer_size;

            thrust::copy(d_W2.begin() + W_row_c, d_W2.begin() + W_row_c + hidden_layer_size, d_temp_W2_row.begin());
            thrust::copy(d_hidden.begin(), d_hidden.begin() + hidden_layer_size, d_temp_X2_row.begin());

            d_logits[c] = thrustnn::dot(d_temp_W2_row, d_temp_X2_row) + d_b2[c];
        }

        // Softmax
        thrustnn::softmax(d_logits, d_probs);

        auto max_it = thrust::max_element(d_probs.begin(), d_probs.end());
        int predicted = max_it - d_probs.begin();
        if (predicted == d_y[s]) correct++;
    }
    float mlp_acc = 100.0f * correct / num_entries;
    printf("1 Hidden Layer MLP Accuracy: %.2f%%\n", mlp_acc);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    printf("MLP GPU time: %.2f seconds\n", ms / 1000.0f);

    if (mlp_acc >= soft_max_acc) {
        printf("MLP was %.2f%% percent more accurate than Softmax\n", mlp_acc - soft_max_acc);
    }
    else {
        printf("Softmax was %.2f%% percent more accurate than MLP\n", soft_max_acc - mlp_acc);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}