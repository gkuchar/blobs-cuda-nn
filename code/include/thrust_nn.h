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
    //TODO: Implement the thrust transformation for dot product
    return 0.0;
}

// y = y + alpha * x   (SAXPY)
inline void saxpy(thrust::device_vector<float>& y,
                  const thrust::device_vector<float>& x,
                  float alpha) {
    //TODO: Implement the thrust transformation for SAXPY
    return;
}

// y = alpha*y
inline void scale(thrust::device_vector<float>& y, float alpha) {
    //TODO: Implement the thrust transformation for scale
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
    //TODO: Implement the thrust transformation for softmax
    return;
}

// Cross-entropy loss for one example given probs and integer label y.
// L = -log(probs[y]) with small epsilon clamp.
inline float cross_entropy_one(const thrust::device_vector<float>& probs, int y) {
    //TODO: Implement the thrust transformation for cross_entropy_one
    return 0.0;
}

} // namespace thrustnn
