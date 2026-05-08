cmake_minimum_required(VERSION 3.20)

project(vectordbfromscratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VDB_BUILD_GRPC "Build gRPC worker/coordinator services when dependencies are available" ON)
option(VDB_BUILD_TESTS "Build local engine tests" ON)

add_library(vdb_core
  src/aligned_vector_store.cpp
  src/distance.cpp
  src/hnsw_index.cpp
)

target_include_directories(vdb_core PUBLIC include)
target_compile_features(vdb_core PUBLIC cxx_std_20)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  add_library(vdb_avx512 OBJECT src/distance_avx512.cpp)
  target_include_directories(vdb_avx512 PRIVATE include)
  target_compile_options(vdb_avx512 PRIVATE -mavx512f)
  target_sources(vdb_core PRIVATE $<TARGET_OBJECTS:vdb_avx512>)
  target_compile_definitions(vdb_core PRIVATE VDB_HAS_AVX512_OBJECT=1)
endif()

add_executable(vdb_demo src/demo.cpp)
target_link_libraries(vdb_demo PRIVATE vdb_core)

if(VDB_BUILD_TESTS)
  enable_testing()
  add_executable(vdb_engine_tests tests/engine_tests.cpp)
  target_link_libraries(vdb_engine_tests PRIVATE vdb_core)
  add_test(NAME vdb_engine_tests COMMAND vdb_engine_tests)
endif()

if(VDB_BUILD_GRPC)
  find_package(Protobuf QUIET)
  find_package(gRPC QUIET)

  if(Protobuf_FOUND AND gRPC_FOUND)
    add_library(vdb_proto proto/vector_db.proto)
    target_link_libraries(vdb_proto PUBLIC protobuf::libprotobuf gRPC::grpc++)
    target_include_directories(vdb_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

    get_target_property(grpc_cpp_plugin_location gRPC::grpc_cpp_plugin LOCATION)
    protobuf_generate(TARGET vdb_proto LANGUAGE cpp)
    protobuf_generate(
      TARGET vdb_proto
      LANGUAGE grpc
      GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
      PLUGIN "protoc-gen-grpc=${grpc_cpp_plugin_location}"
    )

    add_library(vdb_grpc_services
      src/grpc_coordinator_service.cpp
      src/grpc_worker_service.cpp
    )
    target_link_libraries(vdb_grpc_services PUBLIC vdb_core vdb_proto gRPC::grpc++)
    target_include_directories(vdb_grpc_services PUBLIC include ${CMAKE_CURRENT_BINARY_DIR})

    add_executable(vdb_worker src/worker_main.cpp)
    target_link_libraries(vdb_worker PRIVATE vdb_grpc_services)

    add_executable(vdb_coordinator src/coordinator_main.cpp)
    target_link_libraries(vdb_coordinator PRIVATE vdb_grpc_services)
  else()
    message(STATUS "Protobuf/gRPC not found; skipping gRPC targets. Core library and tests still build.")
  endif()
endif()
