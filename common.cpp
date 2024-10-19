#include "common.h"

#include <algorithm>
#include <cstddef>
#include <iostream>


float dot_product_loop(const float* a, const float* b, size_t n) noexcept {
  assert(a && b);

  float res = 0.f;
  for (size_t i = 0; i < n; ++i) {
    res += a[i] * b[i];
  }
  return res;
}

void matmul_loop(const float* a, const float* b, float* c, size_t M, size_t N,
                 size_t K) noexcept {
  assert(a && b && c);
  assert(N > 0 && K > 0 && M > 0);

  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < K; ++j) {
      for (size_t k = 0; k < N; ++k) {
        c[i * K + j] += a[i * N + k] * b[k * K + j];
      }
    }
  }
}


void print_durations(double cpu, double cl) {
  std::cout << "cpu duration: " << cpu 
  << "\ncl duration: " << cl 
  << "\ncl faster by: " << cpu / cl << '\n';
}
