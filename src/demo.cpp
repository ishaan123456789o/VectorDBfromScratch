#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "vdb/distributed.h"
#include "vdb/distance.h"

int main() {
  vdb::HnswConfig config;
  config.dimension = vdb::kDefaultDimension;
  config.metric = vdb::Metric::Cosine;
  config.max_neighbors = 16;
  config.ef_construction = 64;
  config.ef_search = 48;

  vdb::HnswIndex shard_a(config);
  vdb::HnswIndex shard_b(config);

  std::mt19937 rng(42);
  std::normal_distribution<float> normal(0.0F, 1.0F);

  for (std::uint64_t id = 0; id < 200; ++id) {
    std::vector<float> vector(config.dimension);
    for (float& value : vector) {
      value = normal(rng);
    }
    if (id % 2 == 0) {
      shard_a.Add(id, vector);
    } else {
      shard_b.Add(id, vector);
    }
  }

  std::vector<float> query(config.dimension);
  for (float& value : query) {
    value = normal(rng);
  }

  vdb::Coordinator coordinator;
  coordinator.AddShard(std::make_shared<vdb::LocalWorkerShard>("a", &shard_a));
  coordinator.AddShard(std::make_shared<vdb::LocalWorkerShard>("b", &shard_b));

  auto results = coordinator.Search(query, 5);
  std::cout << "distance backend: " << vdb::DistanceBackendName() << '\n';
  for (const auto& result : results) {
    std::cout << result.id << " shard=" << result.shard << " distance=" << result.distance << '\n';
  }
}
