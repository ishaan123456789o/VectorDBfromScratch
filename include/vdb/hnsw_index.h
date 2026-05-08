#pragma once

#include <mutex>
#include <random>
#include <shared_mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "vdb/aligned_vector_store.h"
#include "vdb/types.h"

namespace vdb {

struct HnswConfig {
  std::size_t dimension = kDefaultDimension;
  Metric metric = Metric::Cosine;
  std::size_t max_neighbors = 16;
  std::size_t ef_construction = 96;
  std::size_t ef_search = 64;
  std::uint32_t random_seed = 1337;
};

class HnswIndex {
 public:
  explicit HnswIndex(HnswConfig config = {});
  HnswIndex(const HnswIndex&) = delete;
  HnswIndex& operator=(const HnswIndex&) = delete;
  HnswIndex(HnswIndex&& other) noexcept;
  HnswIndex& operator=(HnswIndex&& other) noexcept;

  void Add(VectorId id, std::span<const float> vector);
  void AddBatch(const std::vector<std::pair<VectorId, std::vector<float>>>& vectors);
  std::vector<SearchResult> Search(std::span<const float> query, std::size_t top_k) const;

  void SaveSnapshot(const std::string& path) const;
  static HnswIndex LoadSnapshot(const std::string& path);

  std::size_t Size() const;
  std::size_t Dimension() const { return store_.Dimension(); }

 private:
  struct Candidate {
    std::size_t index = 0;
    float distance = 0.0F;
  };

  struct Node {
    std::vector<std::vector<std::size_t>> links;
  };

  float Distance(const float* lhs, const float* rhs) const;
  void AddLocked(VectorId id, std::span<const float> vector);
  int RandomLevel();
  std::vector<Candidate> SearchLayer(const float* query, std::size_t entry, std::size_t ef,
                                     int level) const;
  std::vector<Candidate> SelectNeighbors(const std::vector<Candidate>& candidates,
                                         std::size_t limit) const;
  void ConnectMutual(std::size_t lhs, std::size_t rhs, int level);
  void Prune(std::size_t node, int level);

  HnswConfig config_;
  AlignedVectorStore store_;
  std::vector<Node> nodes_;
  std::size_t entry_point_ = 0;
  int max_level_ = -1;
  mutable std::shared_mutex mutex_;
  std::mt19937 rng_;
  std::uniform_real_distribution<float> uniform_{0.0F, 1.0F};
};

}  // namespace vdb
