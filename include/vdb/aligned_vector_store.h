#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "vdb/types.h"

namespace vdb {

class AlignedVectorStore {
 public:
  explicit AlignedVectorStore(std::size_t dimension = kDefaultDimension);
  AlignedVectorStore(const AlignedVectorStore&) = delete;
  AlignedVectorStore& operator=(const AlignedVectorStore&) = delete;
  AlignedVectorStore(AlignedVectorStore&&) noexcept;
  AlignedVectorStore& operator=(AlignedVectorStore&&) noexcept;
  ~AlignedVectorStore();

  std::size_t Add(VectorId id, std::span<const float> vector);
  bool Contains(VectorId id) const;
  std::size_t IndexOf(VectorId id) const;

  const float* GetByIndex(std::size_t index) const;
  float* GetByIndex(std::size_t index);
  VectorId IdAt(std::size_t index) const;
  std::vector<float> CopyVector(std::size_t index) const;

  std::size_t Size() const { return ids_.size(); }
  std::size_t Dimension() const { return dimension_; }
  std::size_t Stride() const { return stride_; }

 private:
  static float* AllocateFloats(std::size_t count);
  void Release();

  std::size_t dimension_ = 0;
  std::size_t stride_ = 0;
  std::size_t capacity_ = 0;
  float* data_ = nullptr;
  std::vector<VectorId> ids_;
  std::unordered_map<VectorId, std::size_t> id_to_index_;
};

}  // namespace vdb
