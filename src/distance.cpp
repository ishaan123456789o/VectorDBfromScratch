#include "vdb/distance.h"

#include <cmath>

namespace vdb {

#if defined(VDB_HAS_AVX512_OBJECT)
float L2DistanceAvx512(const float* lhs, const float* rhs, std::size_t dimension);
float CosineDistanceAvx512(const float* lhs, const float* rhs, std::size_t dimension);
#endif

namespace {

float L2DistanceScalar(const float* lhs, const float* rhs, std::size_t dimension) {
  float sum = 0.0F;
  for (std::size_t i = 0; i < dimension; ++i) {
    const float diff = lhs[i] - rhs[i];
    sum += diff * diff;
  }
  return sum;
}

float CosineDistanceScalar(const float* lhs, const float* rhs, std::size_t dimension) {
  float dot = 0.0F;
  float lhs_norm = 0.0F;
  float rhs_norm = 0.0F;
  for (std::size_t i = 0; i < dimension; ++i) {
    dot += lhs[i] * rhs[i];
    lhs_norm += lhs[i] * lhs[i];
    rhs_norm += rhs[i] * rhs[i];
  }
  if (lhs_norm == 0.0F || rhs_norm == 0.0F) {
    return 1.0F;
  }
  return 1.0F - dot / (std::sqrt(lhs_norm) * std::sqrt(rhs_norm));
}

}  // namespace

bool Avx512Available() {
#if defined(VDB_HAS_AVX512_OBJECT) && (defined(__x86_64__) || defined(_M_X64))
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx512f");
#else
  return true;
#endif
#else
  return false;
#endif
}

const char* DistanceBackendName() {
  return Avx512Available() ? "avx512f" : "scalar";
}

float L2Distance(const float* lhs, const float* rhs, std::size_t dimension) {
#if defined(VDB_HAS_AVX512_OBJECT)
  if (Avx512Available()) {
    return L2DistanceAvx512(lhs, rhs, dimension);
  }
#endif
  return L2DistanceScalar(lhs, rhs, dimension);
}

float CosineDistance(const float* lhs, const float* rhs, std::size_t dimension) {
#if defined(VDB_HAS_AVX512_OBJECT)
  if (Avx512Available()) {
    return CosineDistanceAvx512(lhs, rhs, dimension);
  }
#endif
  return CosineDistanceScalar(lhs, rhs, dimension);
}

}  // namespace vdb
