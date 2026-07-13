# Redline - single-GPU LLM inference engine (C++20 / CUDA).
#
# Targets:
#   redline_core   static library: engine, scheduler, paged-KV allocator, loader, kernels
#   redline_bench  pure-C++ CLI driver for throughput runs (no Python in the measured path)
#   redline        pybind11 Python extension module      (REDLINE_BUILD_PYTHON)
#   redline_tests  googletest unit tests                 (REDLINE_BUILD_TESTS)

cmake_minimum_required(VERSION 3.26)

# Default to both GPUs this project targets: sm_75 (RTX 2060 dev box) and
# sm_89 (RTX 4090 bench pod). Presets narrow to a single arch for faster builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;89")
endif()

project(redline VERSION 0.1.0 LANGUAGES CXX CUDA)

option(REDLINE_BUILD_TESTS "Build redline_tests (googletest)" ON)
option(REDLINE_BUILD_PYTHON "Build the redline Python module (pybind11)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# The static core library links into a Python shared module.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit 12.6 REQUIRED)

include(FetchContent)

# ------------------------------------------------------------- nlohmann/json
# Release tarball instead of a git clone: the json.git history is large and
# the hash pin makes the fetch immutable. v3.11.3.
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

# --------------------------------------------------------------- redline_core
add_library(redline_core STATIC
  src/core/block_allocator.cpp
  src/core/config.cpp
  src/core/engine.cpp
  src/core/model_executor.cpp
  src/core/scheduler.cpp
  src/gemm/cublaslt_gemm.cpp
  src/kernels/argmax.cu
  src/kernels/embed_gather.cu
  src/kernels/kv_gather.cu
  src/kernels/kv_scatter.cu
  src/kernels/paged_attention.cu
  src/kernels/prefill_attention.cu
  src/kernels/rmsnorm.cu
  src/kernels/rope.cu
  src/kernels/silu_mul.cu
  src/loader/convert.cpp
  src/loader/safetensors.cpp
  src/loader/weights.cpp
)
target_include_directories(redline_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(redline_core PUBLIC CUDA::cudart CUDA::cublasLt CUDA::nvtx3)
# PUBLIC: tests/test_config.cpp includes <nlohmann/json.hpp> directly to build
# fixture variants and gets the include path through redline_core.
target_link_libraries(redline_core PUBLIC nlohmann_json::nlohmann_json)
target_compile_options(redline_core PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
  # -lineinfo keeps SASS<->source correlation for Nsight without disabling optimizations.
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
)

# -------------------------------------------------------------- redline_bench
add_executable(redline_bench src/bench_main.cpp)
target_link_libraries(redline_bench PRIVATE redline_core)
target_compile_options(redline_bench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

# -------------------------------------------------------- Python module: redline
if(REDLINE_BUILD_PYTHON)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)

  pybind11_add_module(redline src/api/pybind.cpp)
  target_link_libraries(redline PRIVATE redline_core)
  # Stage the pure-Python client package next to the built module so the
  # single documented `PYTHONPATH=$BUILD` covers both `import redline` and
  # `import redline_client`. A custom target (not POST_BUILD) so client-only
  # .py edits restage even when the module itself has no work to do.
  add_custom_target(redline_client_stage
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_CURRENT_SOURCE_DIR}/python/redline_client
            $<TARGET_FILE_DIR:redline>/redline_client
    COMMENT "Staging redline_client (pure Python) next to the redline module"
    VERBATIM
  )
  add_dependencies(redline redline_client_stage)
endif()

# -------------------------------------------------------------- redline_tests
if(REDLINE_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
  )
  # Match the parent project's CRT on MSVC; no effect elsewhere.
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(redline_tests
    tests/test_allocator.cpp
    tests/test_bf16_convert.cpp
    tests/test_config.cpp
    tests/test_kernels.cu
    tests/test_safetensors.cpp
    tests/test_scheduler.cpp
    tests/test_weights_map.cpp
  )
  target_link_libraries(redline_tests PRIVATE redline_core GTest::gtest_main)

  include(GoogleTest)
  # PRE_TEST: discover cases at ctest time, not build time. The default
  # POST_BUILD mode executes the test binary during the build, which breaks
  # GPU-less builders (e.g. CI) if any CUDA initialization leaks into static
  # scope.
  # Unit tests locate committed data fixtures (the BF16->FP16 golden table,
  # later the RoPE fixture) through REDLINE_TEST_DATA_DIR.
  gtest_discover_tests(redline_tests
    DISCOVERY_MODE PRE_TEST
    PROPERTIES ENVIRONMENT "REDLINE_TEST_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
  )
endif()
