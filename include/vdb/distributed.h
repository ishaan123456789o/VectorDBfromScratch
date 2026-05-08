#pragma once

#include <algorithm>
#include <functional>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vdb/hnsw_index.h"

namespace vdb {

class WorkerShard {
 public:
  virtual ~WorkerShard() = default;
  virtual void Add(VectorId id, std::span<const float> vector) = 0;
  virtual void AddBatch(const std::vector<std::pair<VectorId, std::vector<float>>>& vectors) = 0;
  virtual std::vector<SearchResult> Search(std::span<const float> query, std::size_t top_k) = 0;
};

class LocalWorkerShard final : public WorkerShard {
 public:
  LocalWorkerShard(std::string name, HnswIndex* index) : name_(std::move(name)), index_(index) {}

  void Add(VectorId id, std::span<const float> vector) override {
    index_->Add(id, vector);
  }

  void AddBatch(const std::vector<std::pair<VectorId, std::vector<float>>>& vectors) override {
    index_->AddBatch(vectors);
  }

  std::vector<SearchResult> Search(std::span<const float> query, std::size_t top_k) override {
    auto results = index_->Search(query, top_k);
    for (auto& result : results) {
      result.shard = name_;
    }
    return results;
  }

 private:
  std::string name_;
  HnswIndex* index_ = nullptr;
};

class Coordinator {
 public:
  void AddShard(std::shared_ptr<WorkerShard> shard) { shards_.push_back(std::move(shard)); }

  void Add(VectorId id, std::span<const float> vector) {
    if (shards_.empty()) {
      throw std::runtime_error("cannot add vector without worker shards");
    }
    shards_[ShardFor(id)]->Add(id, vector);
  }

  void AddBatch(const std::vector<std::pair<VectorId, std::vector<float>>>& vectors) {
    if (shards_.empty()) {
      throw std::runtime_error("cannot add vector batch without worker shards");
    }
    std::vector<std::vector<std::pair<VectorId, std::vector<float>>>> by_shard(shards_.size());
    for (const auto& item : vectors) {
      by_shard[ShardFor(item.first)].push_back(item);
    }

    std::vector<std::future<void>> futures;
    for (std::size_t i = 0; i < by_shard.size(); ++i) {
      if (by_shard[i].empty()) {
        continue;
      }
      futures.push_back(std::async(std::launch::async, [&, i] {
        shards_[i]->AddBatch(by_shard[i]);
      }));
    }
    for (auto& future : futures) {
      future.get();
    }
  }

  std::vector<SearchResult> Search(std::span<const float> query, std::size_t top_k) {
    if (shards_.empty() || top_k == 0) {
      return {};
    }
    std::vector<std::future<std::vector<SearchResult>>> futures;
    futures.reserve(shards_.size());
    for (auto& shard : shards_) {
      futures.push_back(std::async(std::launch::async, [&, shard] {
        return shard->Search(query, top_k);
      }));
    }

    std::vector<SearchResult> merged;
    for (auto& future : futures) {
      auto partial = future.get();
      merged.insert(merged.end(), partial.begin(), partial.end());
    }

    std::sort(merged.begin(), merged.end(), BetterResult);
    if (merged.size() > top_k) {
      merged.resize(top_k);
    }
    return merged;
  }

 private:
  std::size_t ShardFor(VectorId id) const {
    return std::hash<VectorId>{}(id) % shards_.size();
  }

  std::vector<std::shared_ptr<WorkerShard>> shards_;
};

}  // namespace vdb
