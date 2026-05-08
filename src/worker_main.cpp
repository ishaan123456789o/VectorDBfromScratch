#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "vdb/grpc_services.h"

int main(int argc, char** argv) {
  const std::string address = argc > 1 ? argv[1] : "0.0.0.0:50051";
  const std::string shard = argc > 2 ? argv[2] : address;

  vdb::HnswConfig config;
  config.dimension = vdb::kDefaultDimension;
  config.metric = vdb::Metric::Cosine;

  vdb::rpc::WorkerService service(shard, config);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  auto server = builder.BuildAndStart();
  std::cout << "worker listening on " << address << '\n';
  server->Wait();
  return 0;
}
