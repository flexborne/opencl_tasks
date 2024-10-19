#include <x86intrin.h>
#include <bits/stdc++.h>

#include <cstdlib>
#include <stdexcept>
#include <cassert>
#include <format>
#include <string_view>

#include <CL/cl.h>

extern cl_context context;
extern cl_command_queue queue;
extern cl_program program;
extern cl_device_id device_id;

constexpr inline auto simd_alignment = 32uz;

void sum_simd(const float* a, const float* b, float* c, size_t n) noexcept;

template <class T>
void sum_loop(const T* a, const T* b, T* c, size_t n) noexcept {
  assert(a && b && c);

  for (size_t i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

[[nodiscard]] float dot_product_loop(const float* a, const float* b, size_t n) noexcept;

// aligned
[[nodiscard]] float dot_product_simd(const float* a, const float* b, size_t n) noexcept;

// M: Number of rows in matrix A (and the number of rows in the result matrix C)
// N: Number of columns in matrix A (and the number of rows in matrix B)
// K: Number of columns in matrix B (and the number of columns in the result matrix C)
void matmul_loop(const float* a, const float* b, float* c, size_t M, size_t N, size_t K) noexcept;

void matmul_simd_u(const float* a, const float* b, float* c, size_t M, size_t N, size_t K) noexcept;

/// @return number of occurences of substring in string
[[nodiscard]] size_t count_loop(std::string_view str, std::string_view substr) noexcept;

// aligned if substr.size() is 32 bytes exactly
// does not support substr with size > 32
[[nodiscard]] size_t count_simd(std::string_view str, std::string_view substr) noexcept;

void print_durations(double cpu_duration, double gpu_duration);