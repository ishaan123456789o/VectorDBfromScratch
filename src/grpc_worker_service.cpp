#include "vdb/grpc_services.h"

#include <exception>

namespace vdb::rpc {
namespace {

vdb::Metric ToCoreMetric(Metric metric) {
  return metric == Metric::L2 ? vdb::Metric::L2 : vdb::Metric::Cosine;
}

void AppendResult(const vdb::SearchResult& source, SearchResponse* response) {
  auto* result = response->add_results();
  result->set_id(source.id);
  result->set_distance(source.distance);
  result->set_shard(source.shard);
}

}  // namespace

WorkerService::WorkerService(std::string shard_name, vdb::HnswConfig config)
    : shard_name_(std::move(shard_name)), index_(config) {}

grpc::Status WorkerService::AddVector(grpc::ServerContext*, const AddVectorRequest* request,
                                      AddVectorResponse* response) {
  try {
    const auto& vector = request->vector();
    index_.Add(vector.id(), {vector.values().data(), static_cast<std::size_t>(vector.values_size())});
    response->set_ok(true);
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, error.what());
  }
}

grpc::Status WorkerService::AddBatch(grpc::ServerContext*, const AddBatchRequest* request,
                                     AddBatchResponse* response) {
  try {
    std::vector<std::pair<vdb::VectorId, std::vector<float>>> vectors;
    vectors.reserve(request->vectors_size());
    for (const auto& vector : request->vectors()) {
      vectors.emplace_back(vector.id(),
                           std::vector<float>(vector.values().begin(), vector.values().end()));
    }
    index_.AddBatch(vectors);
    response->set_inserted(static_cast<std::uint32_t>(vectors.size()));
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, error.what());
  }
}

grpc::Status WorkerService::Search(grpc::ServerContext*, const SearchRequest* request,
                                   SearchResponse* response) {
  try {
    auto results = index_.Search({request->query().data(), static_cast<std::size_t>(request->query_size())},
                                 request->top_k());
    for (auto& result : results) {
      result.shard = shard_name_;
      AppendResult(result, response);
    }
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, error.what());
  }
}

}  // namespace vdb::rpc
