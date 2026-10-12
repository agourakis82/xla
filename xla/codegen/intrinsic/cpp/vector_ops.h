/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_CODEGEN_INTRINSIC_CPP_VECTOR_OPS_H_
#define XLA_CODEGEN_INTRINSIC_CPP_VECTOR_OPS_H_

#if defined(__has_attribute) && __has_attribute(ext_vector_type) && \
    defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "Eigen/Core"

namespace xla {
namespace codegen {
#if defined(__FLT16_MANT_DIG__)
// Half precision (float16)
typedef _Float16 Vec8h __attribute__((vector_size(16)));
typedef _Float16 Vec16h __attribute__((vector_size(32)));
#endif

// Single precision (float32)
typedef float Vec4f __attribute__((vector_size(16)));
typedef float Vec8f __attribute__((vector_size(32)));
typedef float Vec16f __attribute__((vector_size(64)));

// Double precision (float64)
typedef double Vec2d __attribute__((vector_size(16)));
typedef double Vec4d __attribute__((vector_size(32)));
typedef double Vec8d __attribute__((vector_size(64)));

// Corresponding integer types
typedef uint32_t Vec4i __attribute__((vector_size(16)));
typedef uint32_t Vec8i __attribute__((vector_size(32)));
typedef uint32_t Vec16i __attribute__((vector_size(64)));
typedef uint64_t Vec2q __attribute__((vector_size(16)));
typedef uint64_t Vec4q __attribute__((vector_size(32)));
typedef uint64_t Vec8q __attribute__((vector_size(64)));

namespace internal {
template <typename T, bool kIsArithmetic = std::is_arithmetic_v<T>>
struct ElementTypeImpl {
  using Vec = std::remove_cv_t<T>;
  using type = std::remove_reference_t<decltype(Vec{}[0])>;
};

template <typename T>
struct ElementTypeImpl<T, true> {
  using type = std::remove_cv_t<T>;
};

template <typename T>
using ElementType = typename ElementTypeImpl<T>::type;

// Helper type to select the corresponding integer vector type.
template <typename ScalarInt, size_t Width>
struct MakeIntVec;

template <>
struct MakeIntVec<uint32_t, 4> {
  using type = Vec4i;
};
template <>
struct MakeIntVec<uint32_t, 8> {
  using type = Vec8i;
};
template <>
struct MakeIntVec<uint32_t, 16> {
  using type = Vec16i;
};
template <>
struct MakeIntVec<uint64_t, 2> {
  using type = Vec2q;
};
template <>
struct MakeIntVec<uint64_t, 4> {
  using type = Vec4q;
};
template <>
struct MakeIntVec<uint64_t, 8> {
  using type = Vec8q;
};

// This trait takes a float scalar or vector and provides its unsigned integer
// partner.
template <typename FloatVec>
struct CorrespondingIntVector {
 private:
  using ScalarFloat = ElementType<FloatVec>;
  static constexpr size_t kWidth = sizeof(FloatVec) / sizeof(ScalarFloat);
  using ScalarInt =
      std::conditional_t<sizeof(ScalarFloat) == 4, uint32_t, uint64_t>;

 public:
  using type = typename MakeIntVec<ScalarInt, kWidth>::type;
  using scalar_type = ScalarInt;
};

template <>
struct CorrespondingIntVector<float> {
  using type = uint32_t;
  using scalar_type = uint32_t;
};

template <>
struct CorrespondingIntVector<double> {
  using type = uint64_t;
  using scalar_type = uint64_t;
};

template <typename T, typename S, size_t... Is>
constexpr T SplatImpl(S val, std::index_sequence<Is...>) {
  return T{(static_cast<void>(Is), static_cast<ElementType<T>>(val))...};
}
}  // namespace internal

template <typename T>
using ElementType = internal::ElementType<T>;

template <typename T>
inline constexpr size_t NumLanes = sizeof(T) / sizeof(ElementType<T>);

template <typename T, typename S>
constexpr T Splat(S val) {
  if constexpr (std::is_arithmetic_v<T>) {
    return static_cast<T>(val);
  } else {
    return internal::SplatImpl<T>(val, std::make_index_sequence<NumLanes<T>>{});
  }
}

template <typename T>
constexpr ElementType<T> GetLane(T v, size_t idx) {
  if constexpr (std::is_arithmetic_v<T>) {
    (void)idx;
    return v;
  } else {
    return v[idx];
  }
}

template <typename T>
constexpr void SetLane(T& v, size_t idx, ElementType<T> val) {
  if constexpr (std::is_arithmetic_v<T>) {
    (void)idx;
    v = val;
  } else {
    v[idx] = val;
  }
}

template <typename MaskT>
inline bool AnyLane(MaskT mask) {
  if constexpr (std::is_arithmetic_v<MaskT>) {
    return mask != 0;
  } else {
    return __builtin_reduce_or(mask) != 0;
  }
}

// Emits `@llvm.fma` (`a * b + c` with a single rounding).
template <typename T>
inline T Fma(T a, T b, T c) {
  return __builtin_elementwise_fma(a, b, c);
}

// Emits `@llvm.fmuladd` (`a * b + c`).
template <typename T>
inline T Fmuladd(T a, T b, T c) {
#pragma clang fp contract(on)
  return a * b + c;
}

// Contraction-safe rounded product: emits `@llvm.fmuladd(a, b, +0.0)` instead
// of a bare `fmul` so that downstream `SetAllowContractOnFpArithmetic` or
// aggressive-FMA targets cannot fuse this product into a surrounding addition.
template <typename T>
inline T Mul(T a, T b) {
  return Fmuladd(a, b, Splat<T>(0.0));
}

// Evaluates a polynomial c_0 * x^n + c_1 * x^(n-1) + ... + c_n using Horner's
// method with `Fmuladd`, where `coeffs` are given in descending degree order.
template <typename T, typename S, typename... Coeffs>
constexpr T HornerPoly(T x, S c0, Coeffs... coeffs) {
  T acc = Splat<T>(c0);
  ((acc = Fmuladd(acc, x, Splat<T>(coeffs))), ...);
  return acc;
}

// Dekker's error-free transformation `s + e = a + b` assuming `|a| >= |b|`.
template <typename T>
inline std::pair<T, T> FastTwoSum(T a, T b) {
  T s = a + b;
  T e = (a - s) + b;
  return {s, e};
}

// Knuth's error-free transformation `s + e = a + b` without any magnitude
// ordering precondition on `a` and `b`.
template <typename T>
inline std::pair<T, T> TwoSum(T a, T b) {
  T s = a + b;
  T v = s - a;
  T e = (a - (s - v)) + (b - v);
  return {s, e};
}

// ===--------------------------------------------------------------------===//
// Eigen Array type mapping
// ===--------------------------------------------------------------------===//
template <typename VecType>
struct ArrayMap;

#if defined(__FLT16_MANT_DIG__)
template <>
struct ArrayMap<Vec8h> {
  using type = Eigen::Array<Eigen::half, 8, 1>;
};
template <>
struct ArrayMap<Vec16h> {
  using type = Eigen::Array<Eigen::half, 16, 1>;
};
#endif

template <>
struct ArrayMap<Vec4f> {
  using type = Eigen::Array<float, 4, 1>;
};
template <>
struct ArrayMap<Vec8f> {
  using type = Eigen::Array<float, 8, 1>;
};
template <>
struct ArrayMap<Vec16f> {
  using type = Eigen::Array<float, 16, 1>;
};

template <>
struct ArrayMap<Vec2d> {
  using type = Eigen::Array<double, 2, 1>;
};
template <>
struct ArrayMap<Vec4d> {
  using type = Eigen::Array<double, 4, 1>;
};
template <>
struct ArrayMap<Vec8d> {
  using type = Eigen::Array<double, 8, 1>;
};

// Computes the absolute value of a scalar or vector using bitwise operations.
template <typename FloatVec>
FloatVec BitwiseAbs(FloatVec x) {
  using IntVec = typename internal::CorrespondingIntVector<FloatVec>::type;
  using ScalarInt =
      typename internal::CorrespondingIntVector<FloatVec>::scalar_type;
  const IntVec abs_mask = Splat<IntVec>(
      ~(static_cast<ScalarInt>(1) << (sizeof(ScalarInt) * 8 - 1)));
  return __builtin_bit_cast(FloatVec, __builtin_bit_cast(IntVec, x) & abs_mask);
}

// Copies the sign of one scalar or vector to the value of another.
template <typename FloatVec>
FloatVec BitwiseCopysign(FloatVec value, FloatVec sign_source) {
  using IntVec = typename internal::CorrespondingIntVector<FloatVec>::type;
  using ScalarInt =
      typename internal::CorrespondingIntVector<FloatVec>::scalar_type;
  const IntVec sign_mask =
      Splat<IntVec>(static_cast<ScalarInt>(1) << (sizeof(ScalarInt) * 8 - 1));
  FloatVec value_abs = BitwiseAbs<FloatVec>(value);
  IntVec sign_bits = __builtin_bit_cast(IntVec, sign_source) & sign_mask;
  return __builtin_bit_cast(FloatVec,
                            __builtin_bit_cast(IntVec, value_abs) | sign_bits);
}

template <typename Vec, typename Scalar>
Vec Clamp(Vec x, Scalar min, Scalar max) {
  return x < min ? min : x > max ? max : x;
}
}  // namespace codegen
}  // namespace xla

#endif  // defined(__has_attribute) && __has_attribute(ext_vector_type) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

#endif  // XLA_CODEGEN_INTRINSIC_CPP_VECTOR_OPS_H_
