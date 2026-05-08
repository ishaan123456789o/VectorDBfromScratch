#pragma once

#include <memory>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "vdb/hnsw_index.h"
#include "vector_db.grpc.pb.h"

namespace vdb::rpc {

class WorkerService final : public Worker::Service {
 public:
  WorkerService(std::string shard_name, vdb::HnswConfig config);

  grpc::Status AddVector(grpc::ServerContext* context, const AddVectorRequest* request,
                         AddVectorResponse* response) override;
  grpc::Status AddBatch(grpc::ServerContext* context, const AddBatchRequest* request,
                        AddBatchResponse* response) override;
  grpc::Status Search(grpc::ServerContext* context, const SearchRequest* request,
                      SearchResponse* response) override;

 private:
  std::string shard_name_;
  vdb::HnswIndex index_;
};

class CoordinatorService final : public Coordinator::Service {
 public:
  explicit CoordinatorService(std::vector<std::string> worker_addresses);

  grpc::Status AddVector(grpc::ServerContext* context, const AddVectorRequest* request,
                         AddVectorResponse* response) override;
  grpc::Status AddBatch(grpc::ServerContext* context, const AddBatchRequest* request,
                        AddBatchResponse* response) override;
  grpc::Status Search(grpc::ServerContext* context, const SearchRequest* request,
                      SearchResponse* response) override;

 private:
  std::size_t ShardFor(vdb::VectorId id) const;

  std::vector<std::unique_ptr<Worker::Stub>> workers_;
};

}  // namespace vdb::rpc
