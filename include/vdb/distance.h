#pragma once

#include <cstddef>

#include "vdb/types.h"

namespace vdb {

float L2Distance(const float* lhs, const float* rhs, std::size_t dimension);
float CosineDistance(const float* lhs, const float* rhs, std::size_t dimension);

bool Avx512Available();
const char* DistanceBackendName();

}  // namespace vdb
