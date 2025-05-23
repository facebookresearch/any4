// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cuda.h>

// allow usage for non-CUDA files
#ifndef __host__
#define __host__
#define __device__
#endif

namespace tinygemm {

template <typename U, typename V>
constexpr __host__ __device__ auto divDown(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value, "");
  return (a / b);
}

template <typename U, typename V>
constexpr __host__ __device__ auto divUp(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value, "");
  return (a + b - 1) / b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundDown(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value, "");
  return divDown(a, b) * b;
}

template <typename U, typename V>
constexpr __host__ __device__ auto roundUp(U a, V b) -> decltype(a + b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value, "");
  return divUp(a, b) * b;
}

template <typename U, typename V>
constexpr __host__ __device__ bool isEvenDivisor(U a, V b) {
  static_assert(std::is_integral<U>::value && std::is_integral<V>::value, "");
  return (a % V(b) == 0) && ((a / V(b)) >= 1);
}

template <class T>
constexpr __host__ __device__ T pow(T n, int power) {
  return (power > 0 ? n * pow(n, power - 1) : 1);
}

template <class T>
constexpr __host__ __device__ T pow2(int power) {
  return pow(2, power);
}

static_assert(pow2<int>(8) == 256, "pow2");

template <typename T>
constexpr __host__ __device__ int log2(T n, int p = 0) {
  return (n <= 1) ? p : log2(n / 2, p + 1);
}

static_assert(log2(2) == 1, "log2");
static_assert(log2(3) == 1, "log2");
static_assert(log2(4) == 2, "log2");

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v) {
  static_assert(std::is_integral<T>::value, "");
  return (v && !(v & (v - 1)));
}

static_assert(isPowerOf2(2048), "isPowerOf2");
static_assert(!isPowerOf2(3333), "isPowerOf2");

template <typename T>
constexpr __host__ __device__ T nextHighestPowerOf2(T v) {
  static_assert(std::is_integral<T>::value, "");
  return (isPowerOf2(v) ? (T)2 * v : ((T)1 << (log2(v) + 1)));
}

static_assert(nextHighestPowerOf2(1) == 2, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(2) == 4, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(3) == 4, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(4) == 8, "nextHighestPowerOf2");

static_assert(nextHighestPowerOf2(15) == 16, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(16) == 32, "nextHighestPowerOf2");
static_assert(nextHighestPowerOf2(17) == 32, "nextHighestPowerOf2");

static_assert(
    nextHighestPowerOf2(1536000000u) == 2147483648u,
    "nextHighestPowerOf2");
static_assert(
    nextHighestPowerOf2((size_t)2147483648ULL) == (size_t)4294967296ULL,
    "nextHighestPowerOf2");

template <typename T>
constexpr __host__ __device__ T nextLowestPowerOf2(T v) {
  static_assert(std::is_integral<T>::value, "");
  return (isPowerOf2(v) ? v / (T)2 : ((T)1 << (log2(v))));
}

static_assert(nextLowestPowerOf2(1) == 0, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(2) == 1, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(3) == 2, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(4) == 2, "nextLowestPowerOf2");

static_assert(nextLowestPowerOf2(15) == 8, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(16) == 8, "nextLowestPowerOf2");
static_assert(nextLowestPowerOf2(17) == 16, "nextLowestPowerOf2");

inline __host__ __device__ bool isPointerAligned(const void* p, int align) {
  return reinterpret_cast<uintptr_t>(p) % align == 0;
}

// Returns the increment needed to aligned the pointer to the next highest
// aligned address
template <int Align>
inline __host__ __device__ uint32_t getAlignmentRoundUp(const void* p) {
  static_assert(isPowerOf2(Align), "");
  uint32_t diff = uint32_t(uintptr_t(p) & uintptr_t(Align - 1));
  return diff == 0 ? 0 : uint32_t(Align) - diff;
}

} // namespace tinygemm
