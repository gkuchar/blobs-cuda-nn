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

    return 0;
}