# Distributed In-Memory Vector Database from Scratch

This is a resume/learning project that implements the core pieces of a distributed vector database in C++20. I built the storage engine, approximate nearest-neighbor index, SIMD distance kernels, gRPC API surface, and coordinator/worker query architecture from scratch to understand how modern vector databases are put together under the hood.

The project is intentionally small enough to read, but it includes real systems concepts: aligned memory management, HNSW graph search, hardware-aware distance computation, shard-level parallelism, binary persistence snapshots, and protobuf/gRPC service definitions.

## What I Built

- A C++20 in-memory vector storage engine for 768-dimensional embeddings.
- A Hierarchical Navigable Small World (HNSW) graph index for approximate nearest-neighbor search.
- L2 and cosine distance functions with a raw AVX-512 implementation using Intel intrinsics, plus a scalar fallback for machines without AVX-512.
- Strict 64-byte aligned vector buffers using `posix_memalign`, so SIMD loads can use aligned memory.
- A distributed coordinator/worker architecture:
  - workers own shard-local HNSW indexes,
  - the coordinator receives client queries,
  - queries are fanned out to all workers in parallel,
  - shard-local top-k results are merged and globally sorted.
- Coordinator-side write routing using hash partitioning by vector id.
- Batch inserts for local coordinator routing and protobuf/gRPC APIs.
- Binary snapshot save/load for durable recovery of the vector corpus.
- Engine tests covering alignment, distance functions, HNSW search, distributed merge behavior, batch routing, and snapshot restore.

## Architecture

```text
Client
  |
  | Search / AddVector / AddBatch
  v
Coordinator Node
  |
  | fan-out search in parallel
  | hash-routed writes by vector id
  v
Worker Shards
  |
  | each worker owns:
  | - aligned vector storage
  | - HNSW graph index
  | - local top-k search
  v
Final top-k merge at coordinator
```

The coordinator is responsible for distributed query execution. It sends search requests to workers concurrently, collects each shard's nearest neighbors, then sorts and truncates the merged candidate set to produce the final top-k result list.

## Algorithms and Systems Concepts Used

- **HNSW ANN search**: multi-layer proximity graph with greedy descent on upper layers and bounded candidate expansion at the base layer.
- **Approximate nearest neighbor indexing**: trades exact exhaustive scans for faster sublinear retrieval.
- **Cosine distance and L2 distance**: implemented as selectable vector distance metrics.
- **AVX-512 SIMD intrinsics**: uses `_mm512_load_ps`, `_mm512_fmadd_ps`, and related raw intrinsics for accelerated distance computation on supported x86_64 CPUs.
- **64-byte aligned allocation**: vector memory is aligned to cache-line/SIMD boundaries for high-throughput distance kernels.
- **Shard fan-out/fan-in**: distributed search pattern where workers compute local results and a coordinator performs global aggregation.
- **Hash partitioning**: coordinator routes inserts to shards with `hash(vector_id) % shard_count`.
- **Binary snapshot persistence**: saves index configuration and vector payloads to disk, then rebuilds the HNSW graph on load.
- **Concurrent reads**: HNSW uses `std::shared_mutex`, allowing concurrent searches while keeping inserts exclusive.
- **Protocol Buffers and gRPC**: defines worker/coordinator RPC contracts for search and write operations.

## Repository Layout

```text
include/vdb/
  aligned_vector_store.h   64-byte aligned vector storage
  distance.h               distance kernel interface
  distributed.h            local coordinator/worker abstractions
  grpc_services.h          gRPC service declarations
  hnsw_index.h             HNSW index API
  types.h                  shared types

src/
  aligned_vector_store.cpp
  distance.cpp             runtime AVX-512 dispatch + scalar fallback
  distance_avx512.cpp      raw AVX-512 intrinsics
  hnsw_index.cpp           HNSW graph implementation + snapshots
  grpc_worker_service.cpp
  grpc_coordinator_service.cpp
  worker_main.cpp
  coordinator_main.cpp
  demo.cpp

proto/
  vector_db.proto          protobuf/gRPC API

tests/
  engine_tests.cpp         local engine and coordinator tests
```

## Build and Test

This environment does not have `cmake`, gRPC, or protobuf tooling installed, so the repository includes a Makefile for the core engine and local tests:

```sh
make test
make build/vdb_demo
./build/vdb_demo
```

Expected test output:

```text
engine tests passed with backend scalar
```

On an x86_64 machine with AVX-512 support, the distance backend can dispatch to the AVX-512 implementation. On this Mac, the scalar fallback is used.

For a machine with CMake, Protobuf, and gRPC installed:

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

If gRPC/Protobuf are available, CMake builds:

- `vdb_worker`
- `vdb_coordinator`
- gRPC service library and generated protobuf code

## Running the Distributed Services

When gRPC and Protobuf are installed:

```sh
./build/vdb_worker 0.0.0.0:50051 shard-a
./build/vdb_worker 0.0.0.0:50052 shard-b
./build/vdb_coordinator 0.0.0.0:50050 127.0.0.1:50051 127.0.0.1:50052
```

Clients call:

- `vdb.rpc.Coordinator/AddVector`
- `vdb.rpc.Coordinator/AddBatch`
- `vdb.rpc.Coordinator/Search`

The coordinator routes writes to a worker shard and fans search requests out to every worker.

## Current Capabilities

- Insert vectors into an HNSW index.
- Search approximate nearest neighbors by L2 or cosine distance.
- Store vectors in 64-byte aligned memory.
- Use AVX-512 distance kernels on supported x86_64 hardware.
- Run multiple local worker shards through a coordinator.
- Route single and batch writes through the coordinator.
- Save and restore the vector corpus from a binary snapshot.
- Rebuild the HNSW graph after loading a snapshot.
- Expose protobuf/gRPC contracts for distributed deployment.

## What Is Still Not Production-Ready

This project is much closer to a real vector database now, but it is still a learning system rather than production infrastructure.

Remaining gaps:

- No replication or fault-tolerant consensus.
- No automatic cluster membership, discovery, or shard rebalancing.
- Snapshot restore rebuilds the graph instead of storing graph edges directly.
- No deletes, tombstones, compaction, or update-in-place support.
- No metadata payload storage or filtered vector search.
- No authentication, TLS, rate limiting, quotas, or multi-tenancy.
- No benchmark harness for recall, latency percentiles, throughput, and memory usage.
- gRPC targets were not compiled in this local environment because dependencies are missing.

## Good Interview Talking Points

- Why HNSW is faster than brute-force vector search for large corpora.
- How `ef_search`, `ef_construction`, and `max_neighbors` affect recall, latency, and memory use.
- Why vector buffers are 64-byte aligned for AVX-512.
- How cosine distance differs from L2 and where each is useful.
- How fan-out/fan-in distributed search works.
- How shard-local top-k results are merged into a global top-k.
- Why production databases need persistence, replication, metadata filtering, and observability beyond the core ANN algorithm.

## Example Resume Bullet

Built a distributed in-memory vector database in C++20 with an HNSW approximate-nearest-neighbor index, AVX-512 accelerated L2/cosine distance kernels over 768-dimensional embeddings, 64-byte aligned vector storage, protobuf/gRPC worker and coordinator services, parallel fan-out/fan-in search, hash-based shard write routing, batch inserts, and binary snapshot persistence.
