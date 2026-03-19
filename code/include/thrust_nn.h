#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <cmath>
#include <cstdint>

// A small collection of Thrust-based primitives used in ML examples.
// These are intentionally simple and optimized for readability.

namespace thrustnn {

// Dot product: sum_i a[i]*b[i]
inline float dot(const thrust::device_vector<float>& a,
                 const thrust::device_vector<float>& b) {         
    auto first = thrust::make_zip_iterator(thrust::make_tuple(a.begin(), b.begin()));
    auto last  = thrust::make_zip_iterator(thrust::make_tuple(a.end(), b.end()));
    

    return thrust::transform_reduce(first, last,
    [] __device__ (thrust::tuple<float, float> zip) {
        return thrust::get<0>(zip) * thrust::get<1>(zip)
    }, 0.0f, thrust::plus<float>());
}

// y = y + alpha * x   (SAXPY)
inline void saxpy(thrust::device_vector<float>& y,
                  const thrust::device_vector<float>& x,
                  float alpha) {
    thrust::transform(y.begin(), y.end(), x.begin(), y.begin(), 
    [alpha] __device__ (float y_val, float x_val) {
        return x_val * alpha + y_val;
    });
    return;
}

// y = alpha*y
inline void scale(thrust::device_vector<float>& y, float alpha) {
    thrust::transform(y.begin(), y.end(), y.begin(), 
    [alpha] __device__ (float val) {
        return val * alpha;
    });
    return;     
}

inline void sigmoid_inplace(thrust::device_vector<float>& v) {
    //TODO: Implement the thrust transformation for sigmoid_inplace
    return;
}

inline void relu_inplace(thrust::device_vector<float>& v) {
    //TODO: Implement the thrust transformation for relu_inplace
    return;
}

// Stabilized softmax for a vector of length k.
// Input: logits (device), Output: probs (device). probs may alias logits.
inline void softmax(const thrust::device_vector<float>& logits,
                    thrust::device_vector<float>& probs) {
    if (logits.size() == 0) {
        printf("empty logits vector, division by 0 error");
        return;
    }
    if (probs.size() != logits.size()) {
        probs.resize(logits.size());
    }

    float max_logit = thrust::reduce(logits.begin(), logits.end(), std::numeric_limits<float>::lowest(), thrust::maximum<float>());

    thrust::transform(logits.begin(), logits.end(), probs.begin(),
    [max_logit] __device__ (float logit) {
        return expf(logit - max_logit);
    });

    float sum_logits = thrust::reduce(probs.begin(), probs.end(), 0.0f, thrust::plus<float>());

    sum_logits = max(sum_logits, std::numeric_limits<float>::epsilon());

    thrust::transform(probs.begin(), probs.end(), probs.begin(),
    [sum_logits] __device__ (float prob) {
        return prob / sum_logits;
    });

    return;
}

// Cross-entropy loss for one example given probs and integer label y.
// L = -log(probs[y]) with small epsilon clamp.
inline float cross_entropy_one(const thrust::device_vector<float>& probs, int y) {
    float probs_y = max(std::numeric_limits<float>::epsilon(), probs[y]);
    return -logf(probs_y);
}

} // namespace thrustnn
