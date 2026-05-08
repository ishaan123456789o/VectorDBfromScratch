#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace vdb {

inline constexpr std::size_t kDefaultDimension = 768;
inline constexpr std::size_t kVectorAlignment = 64;

using VectorId = std::uint64_t;

enum class Metric {
  L2,
  Cosine,
};

struct SearchResult {
  VectorId id = 0;
  float distance = std::numeric_limits<float>::infinity();
  std::string shard;
};

inline bool BetterResult(const SearchResult& lhs, const SearchResult& rhs) {
  if (lhs.distance != rhs.distance) {
    return lhs.distance < rhs.distance;
  }
  return lhs.id < rhs.id;
}

}  // namespace vdb
