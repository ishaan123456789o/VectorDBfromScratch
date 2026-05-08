#include "vdb/aligned_vector_store.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace vdb {
namespace {

std::size_t RoundUp(std::size_t value, std::size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

}  // namespace

AlignedVectorStore::AlignedVectorStore(std::size_t dimension)
    : dimension_(dimension),
      stride_(RoundUp(dimension, kVectorAlignment / sizeof(float))) {
  if (dimension_ == 0) {
    throw std::invalid_argument("vector dimension must be positive");
  }
}

AlignedVectorStore::AlignedVectorStore(AlignedVectorStore&& other) noexcept {
  *this = std::move(other);
}

AlignedVectorStore& AlignedVectorStore::operator=(AlignedVectorStore&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  Release();
  dimension_ = other.dimension_;
  stride_ = other.stride_;
  capacity_ = other.capacity_;
  data_ = other.data_;
  ids_ = std::move(other.ids_);
  id_to_index_ = std::move(other.id_to_index_);

  other.dimension_ = 0;
  other.stride_ = 0;
  other.capacity_ = 0;
  other.data_ = nullptr;
  return *this;
}

AlignedVectorStore::~AlignedVectorStore() {
  Release();
}

std::size_t AlignedVectorStore::Add(VectorId id, std::span<const float> vector) {
  if (vector.size() != dimension_) {
    throw std::invalid_argument("vector dimension does not match index dimension");
  }
  if (id_to_index_.contains(id)) {
    throw std::invalid_argument("duplicate vector id");
  }

  if (ids_.size() == capacity_) {
    const std::size_t new_capacity = capacity_ == 0 ? 1024 : capacity_ * 2;
    float* new_data = AllocateFloats(new_capacity * stride_);
    if (data_ != nullptr) {
      std::memcpy(new_data, data_, ids_.size() * stride_ * sizeof(float));
      std::free(data_);
    }
    data_ = new_data;
    capacity_ = new_capacity;
  }

  const std::size_t index = ids_.size();
  float* destination = GetByIndex(index);
  std::copy(vector.begin(), vector.end(), destination);
  std::fill(destination + dimension_, destination + stride_, 0.0F);
  ids_.push_back(id);
  id_to_index_.emplace(id, index);
  return index;
}

bool AlignedVectorStore::Contains(VectorId id) const {
  return id_to_index_.contains(id);
}

std::size_t AlignedVectorStore::IndexOf(VectorId id) const {
  auto it = id_to_index_.find(id);
  if (it == id_to_index_.end()) {
    throw std::out_of_range("unknown vector id");
  }
  return it->second;
}

const float* AlignedVectorStore::GetByIndex(std::size_t index) const {
  return data_ + index * stride_;
}

float* AlignedVectorStore::GetByIndex(std::size_t index) {
  return data_ + index * stride_;
}

VectorId AlignedVectorStore::IdAt(std::size_t index) const {
  return ids_.at(index);
}

std::vector<float> AlignedVectorStore::CopyVector(std::size_t index) const {
  const float* vector = GetByIndex(index);
  return {vector, vector + dimension_};
}

float* AlignedVectorStore::AllocateFloats(std::size_t count) {
  void* pointer = nullptr;
  const std::size_t bytes = count * sizeof(float);
  if (posix_memalign(&pointer, kVectorAlignment, bytes) != 0) {
    throw std::bad_alloc();
  }
  std::memset(pointer, 0, bytes);
  return static_cast<float*>(pointer);
}

void AlignedVectorStore::Release() {
  std::free(data_);
  data_ = nullptr;
}

}  // namespace vdb
