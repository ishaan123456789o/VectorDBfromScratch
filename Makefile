CXX ?= c++
CXXFLAGS ?= -std=c++20 -O3 -Wall -Wextra -pedantic -Iinclude
LDFLAGS ?=

CORE_SRCS = src/aligned_vector_store.cpp src/distance.cpp src/hnsw_index.cpp
DEMO_SRCS = $(CORE_SRCS) src/demo.cpp
TEST_SRCS = $(CORE_SRCS) tests/engine_tests.cpp

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  CORE_SRCS += src/distance_avx512.cpp
  CXXFLAGS += -DVDB_HAS_AVX512_OBJECT=1
  AVX512_FLAGS = -mavx512f
endif

.PHONY: all test clean

all: build/vdb_demo build/vdb_engine_tests

build:
	mkdir -p build

build/vdb_demo: $(DEMO_SRCS) | build
	$(CXX) $(CXXFLAGS) $(filter-out src/distance_avx512.cpp,$(DEMO_SRCS)) \
	  $(if $(filter src/distance_avx512.cpp,$(DEMO_SRCS)),$(AVX512_FLAGS) src/distance_avx512.cpp,) \
	  $(LDFLAGS) -o $@

build/vdb_engine_tests: $(TEST_SRCS) | build
	$(CXX) $(CXXFLAGS) $(filter-out src/distance_avx512.cpp,$(TEST_SRCS)) \
	  $(if $(filter src/distance_avx512.cpp,$(TEST_SRCS)),$(AVX512_FLAGS) src/distance_avx512.cpp,) \
	  $(LDFLAGS) -o $@

test: build/vdb_engine_tests
	./build/vdb_engine_tests

clean:
	rm -rf build
