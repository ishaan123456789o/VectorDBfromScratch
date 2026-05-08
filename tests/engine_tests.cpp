#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

#include "vdb/aligned_vector_store.h"
#include "vdb/distributed.h"
#include "vdb/distance.h"

namespace {

std::vector<float> UnitVector(std::size_t dimension, std::size_t hot_index) {
  std::vector<float> vector(dimension, 0.0F);
  vector[hot_index % dimension] = 1.0F;
  return vector;
}

std::vector<float> SmoothVector(std::size_t dimension, float center) {
  std::vector<float> vector(dimension, 0.0F);
  for (std::size_t i = 0; i < dimension; ++i) {
    const float x = static_cast<float>(i) / static_cast<float>(dimension);
    vector[i] = std::exp(-std::fabs(x - center) * 12.0F);
  }
  return vector;
}

void TestAlignedStore() {
  vdb::AlignedVectorStore store(vdb::kDefaultDimension);
  const auto vector = UnitVector(vdb::kDefaultDimension, 7);
  store.Add(42, vector);
  const auto address = reinterpret_cast<std::uintptr_t>(store.GetByIndex(0));
  assert(address % vdb::kVectorAlignment == 0);
  assert(store.Stride() % (vdb::kVectorAlignment / sizeof(float)) == 0);
  assert(store.IdAt(0) == 42);
}

void TestDistance() {
  alignas(64) float lhs[16] = {};
  alignas(64) float rhs[16] = {};
  lhs[0] = 1.0F;
  rhs[0] = 1.0F;
  rhs[1] = 1.0F;
  assert(std::fabs(vdb::L2Distance(lhs, rhs, 16) - 1.0F) < 1e-5F);
  assert(vdb::CosineDistance(lhs, rhs, 16) < 0.3F);
}

void TestHnswFindsInsertedVector() {
  vdb::HnswConfig config;
  config.dimension = vdb::kDefaultDimension;
  config.metric = vdb::Metric::L2;
  config.max_neighbors = 32;
  config.ef_construction = 128;
  config.ef_search = 128;

  vdb::HnswIndex index(config);
  for (std::uint64_t id = 0; id < 128; ++id) {
    index.Add(id, SmoothVector(config.dimension, static_cast<float>(id) / 127.0F));
  }

  const auto query = SmoothVector(config.dimension, 51.0F / 127.0F);
  const auto results = index.Search(query, 5);
  assert(!results.empty());
  assert(results.front().id == 51);
  assert(results.front().distance == 0.0F);
}

void TestCoordinatorMerge() {
  vdb::HnswConfig config;
  config.dimension = vdb::kDefaultDimension;
  config.metric = vdb::Metric::L2;
  config.max_neighbors = 8;
  config.ef_construction = 32;
  config.ef_search = 32;

  vdb::HnswIndex shard_a(config);
  vdb::HnswIndex shard_b(config);
  shard_a.Add(1, UnitVector(config.dimension, 1));
  shard_a.Add(2, UnitVector(config.dimension, 2));
  shard_b.Add(3, UnitVector(config.dimension, 3));
  shard_b.Add(4, UnitVector(config.dimension, 4));

  vdb::Coordinator coordinator;
  coordinator.AddShard(std::make_shared<vdb::LocalWorkerShard>("a", &shard_a));
  coordinator.AddShard(std::make_shared<vdb::LocalWorkerShard>("b", &shard_b));

  const auto query = UnitVector(config.dimension, 3);
  const auto results = coordinator.Search(query, 3);
  assert(results.size() == 3);
  assert(results.front().id == 3);
  assert(results.front().shard == "b");
  for (std::size_t i = 1; i < results.size(); ++i) {
    assert(results[i - 1].distance <= results[i].distance);
  }
}

void TestCoordinatorBatchRouting() {
  vdb::HnswConfig config;
  config.dimension = vdb::kDefaultDimension;
  config.metric = vdb::Metric::L2;
  config.max_neighbors = 8;
  config.ef_construction = 32;
  config.ef_search = 32;

  vdb::HnswIndex shard_a(config);
  vdb::HnswIndex shard_b(config);
  vdb::Coordinator coordinator;
  coordinator.AddShard(std::make_shared<vdb::LocalWorkerShard>("a", &shard_a));
  coordinator.AddShard(std::make_shared<vdb::LocalWorkerShard>("b", &shard_b));

  std::vector<std::pair<vdb::VectorId, std::vector<float>>> vectors;
  vectors.emplace_back(10, UnitVector(config.dimension, 10));
  vectors.emplace_back(11, UnitVector(config.dimension, 11));
  vectors.emplace_back(12, UnitVector(config.dimension, 12));
  vectors.emplace_back(13, UnitVector(config.dimension, 13));
  coordinator.AddBatch(vectors);

  const auto results = coordinator.Search(UnitVector(config.dimension, 12), 1);
  assert(results.size() == 1);
  assert(results.front().id == 12);
  assert(shard_a.Size() + shard_b.Size() == 4);
}

void TestSnapshotRoundTrip() {
  vdb::HnswConfig config;
  config.dimension = vdb::kDefaultDimension;
  config.metric = vdb::Metric::L2;
  config.max_neighbors = 16;
  config.ef_construction = 64;
  config.ef_search = 64;
  config.random_seed = 99;

  vdb::HnswIndex index(config);
  for (std::uint64_t id = 0; id < 32; ++id) {
    index.Add(id, SmoothVector(config.dimension, static_cast<float>(id) / 31.0F));
  }

  const auto path = std::filesystem::temp_directory_path() / "vdb_snapshot_test.bin";
  index.SaveSnapshot(path.string());
  auto restored = vdb::HnswIndex::LoadSnapshot(path.string());
  std::filesystem::remove(path);

  assert(restored.Size() == index.Size());
  const auto results = restored.Search(SmoothVector(config.dimension, 7.0F / 31.0F), 1);
  assert(results.size() == 1);
  assert(results.front().id == 7);
}

}  // namespace

int main() {
  TestAlignedStore();
  TestDistance();
  TestHnswFindsInsertedVector();
  TestCoordinatorMerge();
  TestCoordinatorBatchRouting();
  TestSnapshotRoundTrip();
  std::cout << "engine tests passed with backend " << vdb::DistanceBackendName() << '\n';
}
