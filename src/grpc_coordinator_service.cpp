#include "vdb/grpc_services.h"

#include <algorithm>
#include <future>
#include <stdexcept>
#include <utility>

namespace vdb::rpc {
namespace {

vdb::SearchResult ToCoreResult(const SearchResult& result) {
  return vdb::SearchResult{result.id(), result.distance(), result.shard()};
}

void AppendResult(const vdb::SearchResult& source, SearchResponse* response) {
  auto* result = response->add_results();
  result->set_id(source.id);
  result->set_distance(source.distance);
  result->set_shard(source.shard);
}

}  // namespace

CoordinatorService::CoordinatorService(std::vector<std::string> worker_addresses) {
  workers_.reserve(worker_addresses.size());
  for (const auto& address : worker_addresses) {
    auto channel = grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
    workers_.push_back(Worker::NewStub(channel));
  }
}

std::size_t CoordinatorService::ShardFor(vdb::VectorId id) const {
  if (workers_.empty()) {
    throw std::runtime_error("coordinator has no workers");
  }
  return std::hash<vdb::VectorId>{}(id) % workers_.size();
}

grpc::Status CoordinatorService::AddVector(grpc::ServerContext*, const AddVectorRequest* request,
                                           AddVectorResponse* response) {
  try {
    grpc::ClientContext context;
    AddVectorResponse worker_response;
    const grpc::Status status =
        workers_[ShardFor(request->vector().id())]->AddVector(&context, *request, &worker_response);
    if (!status.ok()) {
      return status;
    }
    response->set_ok(worker_response.ok());
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    return grpc::Status(grpc::StatusCode::UNAVAILABLE, error.what());
  }
}

grpc::Status CoordinatorService::AddBatch(grpc::ServerContext*, const AddBatchRequest* request,
                                          AddBatchResponse* response) {
  try {
    std::vector<AddBatchRequest> shard_requests(workers_.size());
    for (const auto& vector : request->vectors()) {
      *shard_requests[ShardFor(vector.id())].add_vectors() = vector;
    }

    std::vector<std::future<std::uint32_t>> futures;
    for (std::size_t i = 0; i < shard_requests.size(); ++i) {
      if (shard_requests[i].vectors_size() == 0) {
        continue;
      }
      futures.push_back(std::async(std::launch::async, [&, i] {
        grpc::ClientContext context;
        AddBatchResponse shard_response;
        const grpc::Status status = workers_[i]->AddBatch(&context, shard_requests[i], &shard_response);
        if (!status.ok()) {
          throw std::runtime_error(status.error_message());
        }
        return shard_response.inserted();
      }));
    }

    std::uint32_t inserted = 0;
    for (auto& future : futures) {
      inserted += future.get();
    }
    response->set_inserted(inserted);
    return grpc::Status::OK;
  } catch (const std::exception& error) {
    return grpc::Status(grpc::StatusCode::UNAVAILABLE, error.what());
  }
}

grpc::Status CoordinatorService::Search(grpc::ServerContext*, const SearchRequest* request,
                                        SearchResponse* response) {
  std::vector<std::future<std::vector<vdb::SearchResult>>> futures;
  futures.reserve(workers_.size());

  for (auto& worker : workers_) {
    futures.push_back(std::async(std::launch::async, [&worker, request] {
      grpc::ClientContext context;
      SearchResponse shard_response;
      const grpc::Status status = worker->Search(&context, *request, &shard_response);
      if (!status.ok()) {
        throw std::runtime_error(status.error_message());
      }

      std::vector<vdb::SearchResult> results;
      results.reserve(shard_response.results_size());
      for (const auto& result : shard_response.results()) {
        results.push_back(ToCoreResult(result));
      }
      return results;
    }));
  }

  std::vector<vdb::SearchResult> merged;
  try {
    for (auto& future : futures) {
      auto partial = future.get();
      merged.insert(merged.end(), partial.begin(), partial.end());
    }
  } catch (const std::exception& error) {
    return grpc::Status(grpc::StatusCode::UNAVAILABLE, error.what());
  }

  std::sort(merged.begin(), merged.end(), vdb::BetterResult);
  if (merged.size() > request->top_k()) {
    merged.resize(request->top_k());
  }
  for (const auto& result : merged) {
    AppendResult(result, response);
  }
  return grpc::Status::OK;
}

}  // namespace vdb::rpc
