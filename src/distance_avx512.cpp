#include "vdb/distance.h"

#include <immintrin.h>

#include <cmath>

namespace vdb {
namespace {

float HorizontalSum(__m512 value) {
  alignas(64) float lanes[16];
  _mm512_store_ps(lanes, value);
  float sum = 0.0F;
  for (float lane : lanes) {
    sum += lane;
  }
  return sum;
}

}  // namespace

float L2DistanceAvx512(const float* lhs, const float* rhs, std::size_t dimension) {
  __m512 sum = _mm512_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dimension; i += 16) {
    const __m512 a = _mm512_load_ps(lhs + i);
    const __m512 b = _mm512_load_ps(rhs + i);
    const __m512 diff = _mm512_sub_ps(a, b);
    sum = _mm512_fmadd_ps(diff, diff, sum);
  }

  float total = HorizontalSum(sum);
  for (; i < dimension; ++i) {
    const float diff = lhs[i] - rhs[i];
    total += diff * diff;
  }
  return total;
}

float CosineDistanceAvx512(const float* lhs, const float* rhs, std::size_t dimension) {
  __m512 dot = _mm512_setzero_ps();
  __m512 lhs_norm = _mm512_setzero_ps();
  __m512 rhs_norm = _mm512_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= dimension; i += 16) {
    const __m512 a = _mm512_load_ps(lhs + i);
    const __m512 b = _mm512_load_ps(rhs + i);
    dot = _mm512_fmadd_ps(a, b, dot);
    lhs_norm = _mm512_fmadd_ps(a, a, lhs_norm);
    rhs_norm = _mm512_fmadd_ps(b, b, rhs_norm);
  }

  float dot_total = HorizontalSum(dot);
  float lhs_total = HorizontalSum(lhs_norm);
  float rhs_total = HorizontalSum(rhs_norm);
  for (; i < dimension; ++i) {
    dot_total += lhs[i] * rhs[i];
    lhs_total += lhs[i] * lhs[i];
    rhs_total += rhs[i] * rhs[i];
  }
  if (lhs_total == 0.0F || rhs_total == 0.0F) {
    return 1.0F;
  }
  return 1.0F - dot_total / (std::sqrt(lhs_total) * std::sqrt(rhs_total));
}

}  // namespace vdb
