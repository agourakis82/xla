/* Copyright 2026 The OpenXLA Authors.

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

#if defined(__has_attribute) && __has_attribute(ext_vector_type) && \
    defined(__has_builtin) && __has_builtin(__builtin_vectorelements)

#include "xla/codegen/intrinsic/cpp/custom_cpp_intrinsics.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "xla/codegen/intrinsic/cpp/vector_ops.h"

namespace xla::codegen {
namespace {

template <typename Scalar>
struct TrigConstants;

template <>
struct TrigConstants<float> {
  static constexpr float kFastThreshold = 0x1.0p+17f;
  // Adding 1.5 * 2^23 sets the ULP of `j` to 1.0 so round-to-nearest-even
  // rounds `ax * (2/pi)` to an integer `n` and places `n mod 4` in the low 2
  // mantissa bits of `j`.
  static constexpr float kMagic = 0x1.8p+23f;
  static constexpr float kTwoOverPi = 0x1.45f306p-1f;
  // 74-bit split of pi/2 in three non-overlapping float32 constants.
  // Because `Fma(-n, kPiOver2Hi, ax)` is a single-rounding FMA and for
  // `|n| >= 1` both `ax` and `n * kPiOver2Hi` are multiples of 2^-24 with
  // `|ax - n * kPiOver2Hi| < 1`, the first reduction step is exact.
  static constexpr float kPiOver2Hi = 0x1.921fb6p+0f;
  static constexpr float kPiOver2Mid = -0x1.777a5cp-25f;
  static constexpr float kPiOver2Lo = -0x1.ee59dap-50f;

  static constexpr float kSinS1 = -0x1.555556p-3f;
  static constexpr float kSinS1Lo = 0x1.555556p-28f;
  static constexpr float kSinS2 = 0x1.111112p-7f;
  static constexpr float kSinS2Lo = -0x1.dddddep-32f;
  static constexpr float kCosC1 = 0x1.555556p-5f;
  static constexpr float kCosC1Lo = -0x1.555556p-30f;
  static constexpr float kCosC2 = -0x1.6c16c2p-10f;
  static constexpr float kCosC2Lo = 0x1.27d27ep-35f;

  // Degree-13 polynomial coefficients for
  // (sin(r) - r + r^3/6 - r^5/120) / r^7 on [-pi/4, pi/4].
  template <typename T>
  static T EvalSinTailPoly(T r2) {
    return HornerPoly(r2, 0x1.612462p-33f, -0x1.ae6456p-26f, 0x1.71de3ap-19f,
                      -0x1.a01a02p-13f);
  }

  // Degree-14 polynomial coefficients for
  // (cos(r) - 1 + 0.5*r^2 - r^4/24 + r^6/720) / r^8 on [-pi/4, pi/4].
  template <typename T>
  static T EvalCosTailPoly(T r2) {
    return HornerPoly(r2, -0x1.93974ap-37f, 0x1.1eed8ep-29f, -0x1.27e4fcp-22f,
                      0x1.a01a02p-16f);
  }
};

template <>
struct TrigConstants<double> {
  static constexpr double kFastThreshold = 0x1.0p+30;
  // Adding 1.5 * 2^52 sets the ULP of `j` to 1.0 so round-to-nearest-even
  // rounds `ax * (2/pi)` to an integer `n` and places `n mod 4` in the low 2
  // mantissa bits of `j`.
  static constexpr double kMagic = 0x1.8p+52;
  static constexpr double kTwoOverPi = 0x1.45f306dc9c883p-1;
  // 163-bit split of pi/2 in three non-overlapping float64 constants.
  // `kPiOver2Hi` has 3 trailing mantissa zeros so `Fma(-n, kPiOver2Hi, ax)` is
  // exact for `|x| < kFastThreshold`.
  static constexpr double kPiOver2Hi = 0x1.921fb54442d18p+0;
  static constexpr double kPiOver2Mid = 0x1.1a62633145c07p-54;
  static constexpr double kPiOver2Lo = -0x1.f1976b7ed8fbcp-110;
  static constexpr double kSinS1 = -0x1.5555555555555p-3;
  static constexpr double kSinS1Lo = -0x1.5555555555555p-57;
  static constexpr double kSinS2 = 0x1.1111111111111p-7;
  static constexpr double kSinS2Lo = 0x1.1111111111111p-63;
  static constexpr double kCosC1 = 0x1.5555555555555p-5;
  static constexpr double kCosC1Lo = 0x1.5555555555555p-59;
  static constexpr double kCosC2 = -0x1.6c16c16c16c17p-10;
  static constexpr double kCosC2Lo = 0x1.f49f49f49f49fp-65;

  // Degree-19 polynomial coefficients for
  // (sin(r) - r + r^3/6 - r^5/120) / r^7 on [-pi/4, pi/4].
  template <typename T>
  static T EvalSinTailPoly(T r2) {
    return HornerPoly(r2, -0x1.2f49b46814157p-57, 0x1.952c77030ad4ap-49,
                      -0x1.ae7f3e733b81fp-41, 0x1.6124613a86d09p-33,
                      -0x1.ae64567f544e4p-26, 0x1.71de3a556c734p-19,
                      -0x1.a01a01a01a01ap-13);
  }

  // Degree-20 polynomial coefficients for
  // (cos(r) - 1 + 0.5*r^2 - r^4/24 + r^6/720) / r^8 on [-pi/4, pi/4].
  template <typename T>
  static T EvalCosTailPoly(T r2) {
    return HornerPoly(r2, 0x1.e542ba4020225p-62, -0x1.6827863b97d97p-53,
                      0x1.ae7f3e733b81fp-45, -0x1.93974a8c07c9dp-37,
                      0x1.1eed8eff8d898p-29, -0x1.27e4fb7789f5cp-22,
                      0x1.a01a01a01a01ap-16);
  }
};

// 21 words of 2/pi: word 0 is the integer part (0), and words 1..20 hold bits
// [1..1280] after the binary point of 2/pi (`sum_{k=1}^{20} W[k] * 2^{-64*k}`).
static constexpr uint64_t kTwoOverPiTable[21] = {
    0x0000000000000000ULL, 0xa2f9836e4e441529ULL, 0xfc2757d1f534ddc0ULL,
    0xdb6295993c439041ULL, 0xfe5163abdebbc561ULL, 0xb7246e3a424dd2e0ULL,
    0x06492eea09d1921cULL, 0xfe1deb1cb129a73eULL, 0xe88235f52ebb4484ULL,
    0xe99c7026b45f7e41ULL, 0x3991d639835339f4ULL, 0x9c845f8bbdf9283bULL,
    0x1ff897ffde05980fULL, 0xef2f118b5a0a6d1fULL, 0x6d367ecf27cb09b7ULL,
    0x4f463f669e5fea2dULL, 0x7527bac7ebe5f17bULL, 0x3d0739f78a5292eaULL,
    0x6bfb5fb11f8d5d08ULL, 0x56033046fc7b6babULL, 0xf0cfbc209af4361dULL,
};

struct PayneHanekResult {
  uint64_t q_raw;
  double r_hi;
  double r_lo;
};

// Cold-path Payne-Hanek range reduction for a finite double lane
// (`ax >= 2^28`).
inline PayneHanekResult PayneHanekLaneF64(double ax) {
  using uint128_t = unsigned __int128;
  constexpr double kPiOver2Hi = TrigConstants<double>::kPiOver2Hi;
  constexpr double kPiOver2Mid = TrigConstants<double>::kPiOver2Mid;

  uint64_t bits = __builtin_bit_cast(uint64_t, ax);
  int exp = static_cast<int>((bits >> 52) & 0x7ff) - 1023;
  // Shift 53-bit significand left by 2 so that with e = exp + 10, the binary
  // point of the product lies at bit 64 of p2.
  uint64_t m = ((bits & ((1ULL << 52) - 1)) | (1ULL << 52)) << 2;
  int e = exp + 10;
  int idx = e >> 6;
  int shift = e & 63;

  uint64_t w0 = kTwoOverPiTable[idx];
  uint64_t w1 = kTwoOverPiTable[idx + 1];
  uint64_t w2 = kTwoOverPiTable[idx + 2];

  uint64_t hi = (shift == 0) ? w0 : ((w0 << shift) | (w1 >> (64 - shift)));
  uint64_t mid = (shift == 0) ? w1 : ((w1 << shift) | (w2 >> (64 - shift)));

  uint128_t p1 = static_cast<uint128_t>(m) * mid;
  uint128_t p2 = static_cast<uint128_t>(m) * hi + (p1 >> 64);
  uint128_t u128 = (p2 << 64) | static_cast<uint64_t>(p1);
  __int128 s128 = static_cast<__int128>(u128);
  int64_t i_hi = static_cast<int64_t>(s128 >> 76);

  // When fewer than 14 fractional bits cancel, the 2-word product (hi, mid)
  // already provides >= 90 bits of relative precision; otherwise fold in the
  // third word (lo) and low 24 bits (i_lo).
  int64_t top14 = i_hi >> 38;
  double f_hi, f_lo;
  if (__builtin_expect(top14 != 0 && top14 != -1 && top14 != -(1LL << 13), 1)) {
    int64_t i_mid = static_cast<int64_t>((u128 >> 24) & ((1ULL << 52) - 1));
    double d_hi = Mul(static_cast<double>(i_hi), 0x1.0p-52);
    double d_mid = Mul(static_cast<double>(i_mid), 0x1.0p-104);
    auto [fh, fl] = FastTwoSum(d_hi, d_mid);
    f_hi = fh;
    f_lo = fl;
  } else {
    uint64_t w3 = kTwoOverPiTable[idx + 3];
    uint64_t lo = (shift == 0) ? w2 : ((w2 << shift) | (w3 >> (64 - shift)));
    p1 += (static_cast<uint128_t>(m) * lo) >> 64;
    p2 = static_cast<uint128_t>(m) * hi + (p1 >> 64);
    u128 = (p2 << 64) | static_cast<uint64_t>(p1);
    s128 = static_cast<__int128>(u128);
    i_hi = static_cast<int64_t>(s128 >> 76);
    int64_t i_mid = static_cast<int64_t>((u128 >> 24) & ((1ULL << 52) - 1));
    int64_t i_lo = static_cast<int64_t>(u128 & ((1ULL << 24) - 1));
    double d_hi = Mul(static_cast<double>(i_hi), 0x1.0p-52);
    double d_mid = Mul(static_cast<double>(i_mid), 0x1.0p-104);
    double d_lo = Mul(static_cast<double>(i_lo), 0x1.0p-128);
    auto [s1, e1] = FastTwoSum(d_mid, d_lo);
    auto [f0_hi, f0_lo] = FastTwoSum(d_hi, s1);
    auto [fh, fl] = FastTwoSum(f0_hi, f0_lo + e1);
    f_hi = fh;
    f_lo = fl;
  }

  uint64_t q_raw =
      static_cast<uint64_t>((p2 + (uint128_t{1} << 63)) >> 64) & 3ULL;
  double prod_hi = Mul(f_hi, kPiOver2Hi);
  double prod_lo = Fma(f_hi, kPiOver2Hi, -prod_hi);
  prod_lo = Fma(f_lo, kPiOver2Hi, Fma(f_hi, kPiOver2Mid, prod_lo));
  auto [r_hi, r_lo] = FastTwoSum(prod_hi, prod_lo);
  return {q_raw, r_hi, r_lo};
}

// Vectorized cold-path range reduction for float32 (`ax >= 2^17`).
// For `2^17 <= ax < 2^28`, a 3-part Cody-Waite reduction in IEEE-754 `double`
// (`Fma(-n, kPiOver2HiD, ax_d)` is exact for all `ax < 2^52`) provides >= 52
// bits of precision after cancellation using SIMD FP ops; lanes with
// `ax >= 2^28` use `PayneHanekLaneF64`.
template <typename T, typename IntT>
inline void PayneHanekColdF32(T ax, IntT& q_raw, T& r_hi, T& r_lo) {
  constexpr size_t kLanes = NumLanes<T>;
  using U64V = std::conditional_t<kLanes == 1, uint64_t,
                                  uint64_t __attribute__((
                                      vector_size(kLanes * sizeof(uint64_t))))>;
  using F64V =
      std::conditional_t<kLanes == 1, double,
                         double
                         __attribute__((vector_size(kLanes * sizeof(double))))>;

  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kThreshold = TrigConstants<float>::kFastThreshold;

  F64V ax_d;
  if constexpr (kLanes == 1) {
    ax_d = static_cast<double>(ax);
  } else {
    ax_d = __builtin_convertvector(ax, F64V);
  }
  constexpr double kMagicD = TrigConstants<double>::kMagic;
  constexpr double kTwoOverPiD = TrigConstants<double>::kTwoOverPi;
  constexpr double kPiOver2HiD = TrigConstants<double>::kPiOver2Hi;
  constexpr double kPiOver2MidD = TrigConstants<double>::kPiOver2Mid;
  constexpr double kPiOver2LoD = TrigConstants<double>::kPiOver2Lo;

  F64V j = Fma(ax_d, Splat<F64V>(kTwoOverPiD), Splat<F64V>(kMagicD));
  F64V n = j - Splat<F64V>(kMagicD);
  U64V q64 = __builtin_bit_cast(U64V, j) & Splat<U64V>(3ULL);
  F64V t1 = Fma(-n, Splat<F64V>(kPiOver2HiD), ax_d);
  F64V p_hi = Mul(n, Splat<F64V>(kPiOver2MidD));
  F64V p_lo = Fma(n, Splat<F64V>(kPiOver2MidD), -p_hi);
  auto [s, e] = TwoSum(t1, -p_hi);
  F64V r_lo_d = Fma(-n, Splat<F64V>(kPiOver2LoD), e - p_lo);
  auto [r_d, r_d_lo] = FastTwoSum(s, r_lo_d);

  T rf_hi, rf_lo;
  IntT q_cold;
  if constexpr (kLanes == 1) {
    rf_hi = static_cast<float>(r_d);
    rf_lo = static_cast<float>((r_d - static_cast<double>(rf_hi)) + r_d_lo);
    q_cold = static_cast<uint32_t>(q64);
  } else {
    rf_hi = __builtin_convertvector(r_d, T);
    F64V rf_hi_d = __builtin_convertvector(rf_hi, F64V);
    rf_lo = __builtin_convertvector((r_d - rf_hi_d) + r_d_lo, T);
    q_cold = __builtin_convertvector(q64, IntT);
  }

  auto mask = (ax >= Splat<T>(kThreshold)) & (ax < Splat<T>(kInf));
  q_raw = mask ? q_cold : q_raw;
  r_hi = mask ? rf_hi : r_hi;
  r_lo = mask ? rf_lo : r_lo;

  auto need_table = (ax >= Splat<T>(0x1.0p+28f)) & (ax < Splat<T>(kInf));
  if (__builtin_expect(AnyLane(need_table), 0)) {
    IntT q_vec = q_raw;
    T rh_vec = r_hi;
    T rl_vec = r_lo;
    for (size_t lane = 0; lane < kLanes; ++lane) {
      float ax_lane = GetLane(ax, lane);
      if (ax_lane >= 0x1.0p+28f && ax_lane < kInf) {
        auto [q64_lane, rh_d, rl_d] =
            PayneHanekLaneF64(static_cast<double>(ax_lane));
        double r_d_lane = rh_d + rl_d;
        float rf_hi_lane = static_cast<float>(r_d_lane);
        float rf_lo_lane =
            static_cast<float>((rh_d - static_cast<double>(rf_hi_lane)) + rl_d);
        SetLane(q_vec, lane, static_cast<uint32_t>(q64_lane));
        SetLane(rh_vec, lane, rf_hi_lane);
        SetLane(rl_vec, lane, rf_lo_lane);
      }
    }
    q_raw = q_vec;
    r_hi = rh_vec;
    r_lo = rl_vec;
  }
}

// Vector-level cold-path loop marked `noinline, cold` so `LinkIntoModule` /
// `PrepareForInlining` keeps the table lookup out of hot vector code while
// avoiding per-lane `noinline` calls and dynamic-index stack spills.
template <typename T, typename IntT>
__attribute__((noinline, cold)) void PayneHanekColdLoop(T ax, IntT& q_raw,
                                                        T& r_hi, T& r_lo) {
  using Scalar = ElementType<T>;
  if constexpr (std::is_same_v<Scalar, float>) {
    PayneHanekColdF32(ax, q_raw, r_hi, r_lo);
  } else {
    constexpr Scalar kInf = std::numeric_limits<Scalar>::infinity();
    constexpr Scalar kThreshold = TrigConstants<Scalar>::kFastThreshold;
    IntT q_vec = q_raw;
    T rh_vec = r_hi;
    T rl_vec = r_lo;
    for (size_t lane = 0; lane < NumLanes<T>; ++lane) {
      Scalar ax_lane = GetLane(ax, lane);
      if (ax_lane >= kThreshold && ax_lane < kInf) {
        auto [q_lane, rh_lane, rl_lane] = PayneHanekLaneF64(ax_lane);
        SetLane(q_vec, lane, q_lane);
        SetLane(rh_vec, lane, rh_lane);
        SetLane(rl_vec, lane, rl_lane);
      }
    }
    q_raw = q_vec;
    r_hi = rh_vec;
    r_lo = rl_vec;
  }
}

template <typename T, bool kIsCos>
inline T SinCos(T x) {
  using Scalar = ElementType<T>;
  using IntT = typename internal::CorrespondingIntVector<T>::type;
  using ScalarInt = typename internal::CorrespondingIntVector<T>::scalar_type;
  using C = TrigConstants<Scalar>;

  constexpr Scalar kInf = std::numeric_limits<Scalar>::infinity();
  T ax = BitwiseAbs(x);

  // Fast path: magic-number nearest-integer rounding + 3-part Cody-Waite.
  T j = Fma(ax, Splat<T>(C::kTwoOverPi), Splat<T>(C::kMagic));
  T n = j - Splat<T>(C::kMagic);
  IntT q_raw = __builtin_bit_cast(IntT, j) & Splat<IntT>(3);

  T t1 = Fma(-n, Splat<T>(C::kPiOver2Hi), ax);
  T p_hi = Mul(n, Splat<T>(C::kPiOver2Mid));
  T p_lo = Fma(n, Splat<T>(C::kPiOver2Mid), -p_hi);
  auto [s, e] = TwoSum(t1, -p_hi);
  T r_lo = Fma(-n, Splat<T>(C::kPiOver2Lo), e - p_lo);
  auto [r_hi, r_lo_norm] = FastTwoSum(s, r_lo);
  r_lo = r_lo_norm;

  auto need_cold = (ax >= Splat<T>(C::kFastThreshold)) & (ax < Splat<T>(kInf));
  if (__builtin_expect(AnyLane(need_cold), 0)) {
    PayneHanekColdLoop(ax, q_raw, r_hi, r_lo);
  }

  IntT q = (q_raw + Splat<IntT>(kIsCos ? 1 : 0)) & Splat<IntT>(3);

  T r2 = Mul(r_hi, r_hi);
  T e_r2 = Fma(r_hi, r_hi, -r2);
  T r3 = Mul(r2, r_hi);
  T e_r3 = Fma(e_r2, r_hi, Fma(r2, r_hi, -r3));
  T r4 = Mul(r2, r2);
  T e_r4 = Fma(Mul(Splat<T>(2.0), r2), e_r2, Fma(r2, r2, -r4));

  T one_minus_half_r2 = Fma(Splat<T>(-0.5), r2, Splat<T>(1.0));
  T w_lo = Fma(Splat<T>(-0.5), e_r2,
               Fma(Splat<T>(-0.5), r2, Splat<T>(1.0) - one_minus_half_r2));

  T t3_hi = Mul(Splat<T>(C::kSinS1), r3);
  T t3_lo =
      Fma(Splat<T>(C::kSinS1), e_r3,
          Fma(Splat<T>(C::kSinS1Lo), r3, Fma(Splat<T>(C::kSinS1), r3, -t3_hi)));
  auto [s_hi, s_lo] = FastTwoSum(r_hi, t3_hi);

  T c4_hi = Mul(Splat<T>(C::kCosC1), r4);
  T c4_lo =
      Fma(Splat<T>(C::kCosC1), e_r4,
          Fma(Splat<T>(C::kCosC1Lo), r4, Fma(Splat<T>(C::kCosC1), r4, -c4_hi)));
  auto [c_hi, c_lo] = FastTwoSum(one_minus_half_r2, c4_hi);

  T r5 = Mul(r3, r2);
  T e_r5 = Fma(e_r3, r2, Fma(r3, e_r2, Fma(r3, r2, -r5)));
  T t5_hi = Mul(Splat<T>(C::kSinS2), r5);
  T t5_lo =
      Fma(Splat<T>(C::kSinS2), e_r5,
          Fma(Splat<T>(C::kSinS2Lo), r5, Fma(Splat<T>(C::kSinS2), r5, -t5_hi)));
  auto [s2_hi, s2_lo] = FastTwoSum(s_hi, t5_hi);

  T r6 = Mul(r4, r2);
  T e_r6 = Fma(e_r4, r2, Fma(r4, e_r2, Fma(r4, r2, -r6)));
  T c6_hi = Mul(Splat<T>(C::kCosC2), r6);
  T c6_lo =
      Fma(Splat<T>(C::kCosC2), e_r6,
          Fma(Splat<T>(C::kCosC2Lo), r6, Fma(Splat<T>(C::kCosC2), r6, -c6_hi)));
  auto [c2_hi, c2_lo] = FastTwoSum(c_hi, c6_hi);

  T p_sin = C::EvalSinTailPoly(r2);
  T r7 = Mul(r5, r2);
  T e_r7 = Fma(e_r5, r2, Fma(r5, e_r2, Fma(r5, r2, -r7)));
  T t7_hi = Mul(r7, p_sin);
  T t7_lo = Fma(e_r7, p_sin, Fma(r7, p_sin, -t7_hi));
  auto [s3_hi, s3_lo] = FastTwoSum(s2_hi, t7_hi);
  T sin_val = s3_hi + (((s_lo + s2_lo) + s3_lo) +
                       ((t3_lo + (t5_lo + t7_lo)) + Mul(r_lo, c2_hi)));

  T p_cos = C::EvalCosTailPoly(r2);
  T r8 = Mul(r4, r4);
  T e_r8 = Fma(Mul(Splat<T>(2.0), r4), e_r4, Fma(r4, r4, -r8));
  T c8_hi = Mul(r8, p_cos);
  T c8_lo = Fma(e_r8, p_cos, Fma(r8, p_cos, -c8_hi));
  auto [c3_hi, c3_lo] = FastTwoSum(c2_hi, c8_hi);
  T cos_val = c3_hi + (((c_lo + c2_lo) + (c3_lo + w_lo)) +
                       ((c4_lo - Mul(s2_hi, r_lo)) + (c6_lo + c8_lo)));

  T y = ((q & Splat<IntT>(1)) != Splat<IntT>(0)) ? cos_val : sin_val;

  // Bit 1 of `q` is the quadrant sign (`q in {2, 3}`); shift left by `bits - 2`
  // to move it into the IEEE-754 sign bit (`bits - 1`).
  constexpr size_t kSignShift = sizeof(ScalarInt) * 8 - 2;
  IntT sign_bit = (q & Splat<IntT>(2)) << kSignShift;
  if constexpr (!kIsCos) {
    constexpr ScalarInt kSignMask = static_cast<ScalarInt>(1)
                                    << (sizeof(ScalarInt) * 8 - 1);
    sign_bit ^= (__builtin_bit_cast(IntT, x) & Splat<IntT>(kSignMask));
  }
  y = __builtin_bit_cast(T, __builtin_bit_cast(IntT, y) ^ sign_bit);

  return (ax < Splat<T>(kInf)) ? y : (x - x);
}

}  // namespace

float sin_f32(float x) { return SinCos<float, false>(x); }
Vec4f sin_v4f32(Vec4f x) { return SinCos<Vec4f, false>(x); }
Vec8f sin_v8f32(Vec8f x) { return SinCos<Vec8f, false>(x); }
Vec16f sin_v16f32(Vec16f x) { return SinCos<Vec16f, false>(x); }
double sin_f64(double x) { return SinCos<double, false>(x); }
Vec2d sin_v2f64(Vec2d x) { return SinCos<Vec2d, false>(x); }
Vec4d sin_v4f64(Vec4d x) { return SinCos<Vec4d, false>(x); }
Vec8d sin_v8f64(Vec8d x) { return SinCos<Vec8d, false>(x); }

float cos_f32(float x) { return SinCos<float, true>(x); }
Vec4f cos_v4f32(Vec4f x) { return SinCos<Vec4f, true>(x); }
Vec8f cos_v8f32(Vec8f x) { return SinCos<Vec8f, true>(x); }
Vec16f cos_v16f32(Vec16f x) { return SinCos<Vec16f, true>(x); }
double cos_f64(double x) { return SinCos<double, true>(x); }
Vec2d cos_v2f64(Vec2d x) { return SinCos<Vec2d, true>(x); }
Vec4d cos_v4f64(Vec4d x) { return SinCos<Vec4d, true>(x); }
Vec8d cos_v8f64(Vec8d x) { return SinCos<Vec8d, true>(x); }

}  // namespace xla::codegen

#endif  // defined(__has_attribute) && __has_attribute(ext_vector_type) &&
        // defined(__has_builtin) && __has_builtin(__builtin_vectorelements)
