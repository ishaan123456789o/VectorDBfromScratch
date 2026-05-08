#include <iostream>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "vdb/grpc_services.h"

int main(int argc, char** argv) {
  const std::string address = argc > 1 ? argv[1] : "0.0.0.0:50050";
  std::vector<std::string> workers;
  for (int i = 2; i < argc; ++i) {
    workers.emplace_back(argv[i]);
  }
  if (workers.empty()) {
    workers.emplace_back("127.0.0.1:50051");
  }

  vdb::rpc::CoordinatorService service(workers);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  auto server = builder.BuildAndStart();
  std::cout << "coordinator listening on " << address << " with " << workers.size()
            << " workers\n";
  server->Wait();
  return 0;
}
