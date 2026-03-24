#include <iostream>
#include <random>
#include <chrono>
#include "../include/data_io.h"

inline float dot(const std::vector<float>& a, const std::vector<float>& b) {      
    if (a.size() != b.size()) {
        printf("Error: vectors must be equal length\n");
        return -1.0f;
    }   
    float ret = 0;
    for (int i = 0; i < a.size(); i++) {
        ret += a[i] * b[i];
    }
    return ret;
}

inline void saxpy(std::vector<float>& y, const std::vector<float>& x, float alpha) {
    if (y.size() != x.size()) {
        printf("Error: vectors must be equal length\n");
        return;
    }   
    for (int i = 0; i < y.size(); i++) {
        y[i] = y[i] + alpha * x[i];
    }
}

inline void softmax(const std::vector<float>& logits, std::vector<float>& probs) {
    if (logits.size() == 0) {
        printf("empty logits vector, division by 0 error");
        return;
    }
    if (probs.size() != logits.size()) {
        probs.resize(logits.size());
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < logits.size(); i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    for (int i = 0; i < logits.size(); i++) {
        probs[i] = expf(logits[i] - max_logit);
    }

    float sum_probs = 0.0f;
    for (int i = 0; i < probs.size(); i++) {
        sum_probs += probs[i];
    }
    sum_probs = std::max(sum_probs, std::numeric_limits<float>::epsilon());

    for (int i = 0; i < logits.size(); i++) {
        probs[i] = probs[i] / sum_probs;
    }

    return;
}

inline float cross_entropy_one(const std::vector<float>& probs, int y) {
    float probs_y = std::max(std::numeric_limits<float>::epsilon(), probs[y]);
    return -logf(probs_y);
}

int main(int argc, char* argv[]) {
    // CPU Stabilized Softmax Classifier with Timings
    // a. Load data onto device: input matrix X, target vector y

    std::vector<float> X;
    std::vector<int> y;

    int num_entries = 0;
    int num_parameters = 0;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <dataset.csv> <num_epochs>\n";
        return 1;
    }

    bool completed = dataio::load_csv_features_labels(argv[1], X, y, num_entries, num_parameters);

    if (!completed) {
        printf("Error loading data\n");
        exit(1);
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();

    // b. Initalize hyperparameters + Weights and bias vectors
    float learning_rate = 0.1f;

    int num_epochs = atoi(argv[2]);
    if (num_epochs <= 0 || num_epochs > 10000) {
        std::cerr << "Epoch count must be between 1 and 10000\n";
        return 1;
    }

    int batch_size = 32;
    int num_classes = 3;

    // Weights and bias vectors
    std::vector<float> W(num_parameters * num_classes);
    std::vector<float> b(num_classes);

    // Fill init with random but realistic values
    std::random_device rd;  // Seed
    std::mt19937 gen(rd()); // Standard generator
    std::normal_distribution<float> dist(0.0f, 0.01f);

    for (int i = 0; i < W.size(); i++) {
        W[i] = dist(gen);
    }

    for (int i = 0; i < b.size(); i++) {
        b[i] = dist(gen);
    }


    // logits and probs vectors
    std::vector<float> logits(num_classes, 0.0f);
    std::vector<float> probs(num_classes, 0.0f);

    // Temp row vectors for logit computation
    std::vector<float> temp_W_row(num_parameters);
    std::vector<float> temp_X_row(num_parameters);

    // c. Training Loop
    int num_batches = num_entries / batch_size;   
    int batch_start, batch_end, W_row_c, W_row_h;
    float loss, error_c, cost;
    for (int i = 0; i < num_epochs; i++) {
        cost = 0;
        for (int j = 0; j < num_batches; j++) {
            batch_start = j * batch_size;
            batch_end = std::min(batch_start + batch_size, num_entries);
            for (int s = batch_start; s < batch_end; s++) {

                // Calculate logit and probability of each class
                for (int c = 0; c < num_classes; c++) {
                    W_row_c = c * num_parameters;
    
                    for (int j = 0; j < temp_W_row.size(); j++) {
                        temp_W_row[j] = W[W_row_c + j];
                        temp_X_row[j] = X[(s * num_parameters) + j];
                    }
        
                    logits[c] = dot(temp_W_row, temp_X_row) + b[c];
                }
                softmax(logits, probs);

                // Calculate loss and accumlate into cost
                loss = cross_entropy_one(probs, y[s]);
                cost += loss;

                // Compute error, update row W and b
                for (int c = 0; c < num_classes; c++) {
                    W_row_c = c * num_parameters;

                    for (int j = 0; j < num_parameters; j++) {
                        temp_W_row[j] = W[W_row_c + j];
                    }
                    error_c = (float)probs[c] - (c == y[s] ? 1 : 0);
                    saxpy(temp_W_row, temp_X_row, -learning_rate * error_c);
                    for (int j = 0; j < num_parameters; j++) {
                        W[W_row_c + j] = temp_W_row[j];
                    }

                    b[c] = b[c] - learning_rate * error_c;
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

            for (int j = 0; j < temp_W_row.size(); j++) {
                temp_W_row[j] = W[W_row_c + j];
                temp_X_row[j] = X[(s * num_parameters) + j];
            }
            logits[c] = dot(temp_W_row, temp_X_row) + b[c];
        }
        softmax(logits, probs);

        int predicted = -1;
        float best = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < probs.size(); j ++) {
            if (probs[j] > best) {
                best = probs[j];
                predicted = j;
            }
        }
        if (predicted == y[s]) correct++;
    }
    float soft_max_acc = 100.0f * correct / num_entries;
    printf("CPU Softmax Accuracy: %.2f%%\n", soft_max_acc);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_ms = cpu_end - cpu_start;

    printf("Dataset: %s\n", argv[1]);
    printf("CPU Softmax time: %.2f ms\n", cpu_ms.count());

    return 0;
}