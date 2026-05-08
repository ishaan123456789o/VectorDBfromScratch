#include "vdb/hnsw_index.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include "vdb/distance.h"

namespace vdb {
namespace {

constexpr std::uint32_t kSnapshotMagic = 0x31424456U;
constexpr std::uint32_t kSnapshotVersion = 1;

struct MinCandidate {
  std::size_t index = 0;
  float distance = 0.0F;
  bool operator<(const MinCandidate& other) const {
    return distance > other.distance;
  }
};

struct MaxCandidate {
  std::size_t index = 0;
  float distance = 0.0F;
  bool operator<(const MaxCandidate& other) const {
    return distance < other.distance;
  }
};

}  // namespace

HnswIndex::HnswIndex(HnswConfig config)
    : config_(config), store_(config.dimension), rng_(config.random_seed) {
  if (config_.max_neighbors == 0 || config_.ef_construction == 0 || config_.ef_search == 0) {
    throw std::invalid_argument("HNSW neighbor and ef parameters must be positive");
  }
}

HnswIndex::HnswIndex(HnswIndex&& other) noexcept
    : config_(other.config_),
      store_(std::move(other.store_)),
      nodes_(std::move(other.nodes_)),
      entry_point_(other.entry_point_),
      max_level_(other.max_level_),
      rng_(std::move(other.rng_)),
      uniform_(other.uniform_) {
  other.entry_point_ = 0;
  other.max_level_ = -1;
}

HnswIndex& HnswIndex::operator=(HnswIndex&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  std::unique_lock lhs_lock(mutex_, std::defer_lock);
  std::unique_lock rhs_lock(other.mutex_, std::defer_lock);
  std::lock(lhs_lock, rhs_lock);

  config_ = other.config_;
  store_ = std::move(other.store_);
  nodes_ = std::move(other.nodes_);
  entry_point_ = other.entry_point_;
  max_level_ = other.max_level_;
  rng_ = std::move(other.rng_);
  uniform_ = other.uniform_;

  other.entry_point_ = 0;
  other.max_level_ = -1;
  return *this;
}

void HnswIndex::Add(VectorId id, std::span<const float> vector) {
  std::unique_lock lock(mutex_);
  AddLocked(id, vector);
}

void HnswIndex::AddBatch(const std::vector<std::pair<VectorId, std::vector<float>>>& vectors) {
  std::unique_lock lock(mutex_);
  for (const auto& [id, vector] : vectors) {
    AddLocked(id, vector);
  }
}

void HnswIndex::AddLocked(VectorId id, std::span<const float> vector) {
  if (vector.size() != config_.dimension) {
    throw std::invalid_argument("vector dimension does not match index dimension");
  }

  const int level = RandomLevel();
  const std::size_t new_index = store_.Add(id, vector);
  nodes_.push_back(Node{std::vector<std::vector<std::size_t>>(static_cast<std::size_t>(level + 1))});

  if (nodes_.size() == 1) {
    entry_point_ = new_index;
    max_level_ = level;
    return;
  }

  const float* new_vector = store_.GetByIndex(new_index);
  std::size_t current = entry_point_;
  float current_distance = Distance(new_vector, store_.GetByIndex(current));

  for (int layer = max_level_; layer > level; --layer) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (std::size_t neighbor : nodes_[current].links[static_cast<std::size_t>(layer)]) {
        const float candidate_distance = Distance(new_vector, store_.GetByIndex(neighbor));
        if (candidate_distance < current_distance) {
          current = neighbor;
          current_distance = candidate_distance;
          changed = true;
        }
      }
    }
  }

  for (int layer = std::min(level, max_level_); layer >= 0; --layer) {
    auto candidates = SearchLayer(new_vector, current, config_.ef_construction, layer);
    auto selected = SelectNeighbors(candidates, config_.max_neighbors);
    for (const auto& candidate : selected) {
      ConnectMutual(new_index, candidate.index, layer);
    }
    if (!selected.empty()) {
      current = selected.front().index;
    }
  }

  if (level > max_level_) {
    entry_point_ = new_index;
    max_level_ = level;
  }
}

std::vector<SearchResult> HnswIndex::Search(std::span<const float> query, std::size_t top_k) const {
  std::shared_lock lock(mutex_);
  if (query.size() != config_.dimension) {
    throw std::invalid_argument("query dimension does not match index dimension");
  }
  if (top_k == 0 || nodes_.empty()) {
    return {};
  }

  std::size_t current = entry_point_;
  float current_distance = Distance(query.data(), store_.GetByIndex(current));

  for (int layer = max_level_; layer > 0; --layer) {
    bool changed = true;
    while (changed) {
      changed = false;
      for (std::size_t neighbor : nodes_[current].links[static_cast<std::size_t>(layer)]) {
        const float candidate_distance = Distance(query.data(), store_.GetByIndex(neighbor));
        if (candidate_distance < current_distance) {
          current = neighbor;
          current_distance = candidate_distance;
          changed = true;
        }
      }
    }
  }

  const std::size_t ef = std::max(config_.ef_search, top_k);
  auto candidates = SearchLayer(query.data(), current, ef, 0);
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& lhs, const Candidate& rhs) {
    if (lhs.distance != rhs.distance) {
      return lhs.distance < rhs.distance;
    }
    return lhs.index < rhs.index;
  });

  std::vector<SearchResult> results;
  results.reserve(std::min(top_k, candidates.size()));
  for (const auto& candidate : candidates) {
    if (results.size() == top_k) {
      break;
    }
    results.push_back(SearchResult{store_.IdAt(candidate.index), candidate.distance, {}});
  }
  return results;
}

void HnswIndex::SaveSnapshot(const std::string& path) const {
  std::shared_lock lock(mutex_);
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open snapshot for writing");
  }

  const auto metric = static_cast<std::uint32_t>(config_.metric);
  const auto dimension = static_cast<std::uint64_t>(config_.dimension);
  const auto max_neighbors = static_cast<std::uint64_t>(config_.max_neighbors);
  const auto ef_construction = static_cast<std::uint64_t>(config_.ef_construction);
  const auto ef_search = static_cast<std::uint64_t>(config_.ef_search);
  const auto random_seed = static_cast<std::uint32_t>(config_.random_seed);
  const auto size = static_cast<std::uint64_t>(store_.Size());

  output.write(reinterpret_cast<const char*>(&kSnapshotMagic), sizeof(kSnapshotMagic));
  output.write(reinterpret_cast<const char*>(&kSnapshotVersion), sizeof(kSnapshotVersion));
  output.write(reinterpret_cast<const char*>(&metric), sizeof(metric));
  output.write(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
  output.write(reinterpret_cast<const char*>(&max_neighbors), sizeof(max_neighbors));
  output.write(reinterpret_cast<const char*>(&ef_construction), sizeof(ef_construction));
  output.write(reinterpret_cast<const char*>(&ef_search), sizeof(ef_search));
  output.write(reinterpret_cast<const char*>(&random_seed), sizeof(random_seed));
  output.write(reinterpret_cast<const char*>(&size), sizeof(size));

  for (std::size_t i = 0; i < store_.Size(); ++i) {
    const VectorId id = store_.IdAt(i);
    output.write(reinterpret_cast<const char*>(&id), sizeof(id));
    output.write(reinterpret_cast<const char*>(store_.GetByIndex(i)),
                 static_cast<std::streamsize>(config_.dimension * sizeof(float)));
  }
  if (!output) {
    throw std::runtime_error("failed while writing snapshot");
  }
}

HnswIndex HnswIndex::LoadSnapshot(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open snapshot for reading");
  }

  std::uint32_t magic = 0;
  std::uint32_t version = 0;
  std::uint32_t metric = 0;
  std::uint64_t dimension = 0;
  std::uint64_t max_neighbors = 0;
  std::uint64_t ef_construction = 0;
  std::uint64_t ef_search = 0;
  std::uint32_t random_seed = 0;
  std::uint64_t size = 0;

  input.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  input.read(reinterpret_cast<char*>(&version), sizeof(version));
  input.read(reinterpret_cast<char*>(&metric), sizeof(metric));
  input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
  input.read(reinterpret_cast<char*>(&max_neighbors), sizeof(max_neighbors));
  input.read(reinterpret_cast<char*>(&ef_construction), sizeof(ef_construction));
  input.read(reinterpret_cast<char*>(&ef_search), sizeof(ef_search));
  input.read(reinterpret_cast<char*>(&random_seed), sizeof(random_seed));
  input.read(reinterpret_cast<char*>(&size), sizeof(size));
  if (!input || magic != kSnapshotMagic || version != kSnapshotVersion) {
    throw std::runtime_error("invalid or unsupported snapshot");
  }

  HnswConfig config;
  config.metric = metric == static_cast<std::uint32_t>(Metric::L2) ? Metric::L2 : Metric::Cosine;
  config.dimension = static_cast<std::size_t>(dimension);
  config.max_neighbors = static_cast<std::size_t>(max_neighbors);
  config.ef_construction = static_cast<std::size_t>(ef_construction);
  config.ef_search = static_cast<std::size_t>(ef_search);
  config.random_seed = random_seed;

  HnswIndex index(config);
  std::vector<float> vector(config.dimension);
  for (std::uint64_t i = 0; i < size; ++i) {
    VectorId id = 0;
    input.read(reinterpret_cast<char*>(&id), sizeof(id));
    input.read(reinterpret_cast<char*>(vector.data()),
               static_cast<std::streamsize>(vector.size() * sizeof(float)));
    if (!input) {
      throw std::runtime_error("truncated snapshot");
    }
    index.Add(id, vector);
  }
  return index;
}

std::size_t HnswIndex::Size() const {
  std::shared_lock lock(mutex_);
  return store_.Size();
}

float HnswIndex::Distance(const float* lhs, const float* rhs) const {
  switch (config_.metric) {
    case Metric::L2:
      return L2Distance(lhs, rhs, config_.dimension);
    case Metric::Cosine:
      return CosineDistance(lhs, rhs, config_.dimension);
  }
  return L2Distance(lhs, rhs, config_.dimension);
}

int HnswIndex::RandomLevel() {
  int level = 0;
  const float probability = 1.0F / static_cast<float>(config_.max_neighbors);
  while (uniform_(rng_) < probability && level < 32) {
    ++level;
  }
  return level;
}

std::vector<HnswIndex::Candidate> HnswIndex::SearchLayer(const float* query, std::size_t entry,
                                                         std::size_t ef, int level) const {
  std::priority_queue<MinCandidate> candidates;
  std::priority_queue<MaxCandidate> nearest;
  std::unordered_set<std::size_t> visited;
  visited.reserve(ef * 4);

  const float entry_distance = Distance(query, store_.GetByIndex(entry));
  candidates.push(MinCandidate{entry, entry_distance});
  nearest.push(MaxCandidate{entry, entry_distance});
  visited.insert(entry);

  while (!candidates.empty()) {
    const auto candidate = candidates.top();
    candidates.pop();
    if (!nearest.empty() && candidate.distance > nearest.top().distance) {
      break;
    }

    const auto& links = nodes_[candidate.index].links[static_cast<std::size_t>(level)];
    for (std::size_t neighbor : links) {
      if (!visited.insert(neighbor).second) {
        continue;
      }
      const float distance = Distance(query, store_.GetByIndex(neighbor));
      if (nearest.size() < ef || distance < nearest.top().distance) {
        candidates.push(MinCandidate{neighbor, distance});
        nearest.push(MaxCandidate{neighbor, distance});
        if (nearest.size() > ef) {
          nearest.pop();
        }
      }
    }
  }

  std::vector<Candidate> result;
  result.reserve(nearest.size());
  while (!nearest.empty()) {
    result.push_back(Candidate{nearest.top().index, nearest.top().distance});
    nearest.pop();
  }
  std::sort(result.begin(), result.end(), [](const Candidate& lhs, const Candidate& rhs) {
    return lhs.distance < rhs.distance;
  });
  return result;
}

std::vector<HnswIndex::Candidate> HnswIndex::SelectNeighbors(
    const std::vector<Candidate>& candidates, std::size_t limit) const {
  std::vector<Candidate> selected = candidates;
  std::sort(selected.begin(), selected.end(), [](const Candidate& lhs, const Candidate& rhs) {
    if (lhs.distance != rhs.distance) {
      return lhs.distance < rhs.distance;
    }
    return lhs.index < rhs.index;
  });
  if (selected.size() > limit) {
    selected.resize(limit);
  }
  return selected;
}

void HnswIndex::ConnectMutual(std::size_t lhs, std::size_t rhs, int level) {
  auto& lhs_links = nodes_[lhs].links[static_cast<std::size_t>(level)];
  auto& rhs_links = nodes_[rhs].links[static_cast<std::size_t>(level)];
  if (std::find(lhs_links.begin(), lhs_links.end(), rhs) == lhs_links.end()) {
    lhs_links.push_back(rhs);
  }
  if (std::find(rhs_links.begin(), rhs_links.end(), lhs) == rhs_links.end()) {
    rhs_links.push_back(lhs);
  }
  Prune(lhs, level);
  Prune(rhs, level);
}

void HnswIndex::Prune(std::size_t node, int level) {
  auto& links = nodes_[node].links[static_cast<std::size_t>(level)];
  if (links.size() <= config_.max_neighbors) {
    return;
  }
  const float* vector = store_.GetByIndex(node);
  std::sort(links.begin(), links.end(), [&](std::size_t lhs, std::size_t rhs) {
    const float lhs_distance = Distance(vector, store_.GetByIndex(lhs));
    const float rhs_distance = Distance(vector, store_.GetByIndex(rhs));
    if (lhs_distance != rhs_distance) {
      return lhs_distance < rhs_distance;
    }
    return lhs < rhs;
  });
  links.resize(config_.max_neighbors);
}

}  // namespace vdb
