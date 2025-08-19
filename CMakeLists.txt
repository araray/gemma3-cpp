cmake_minimum_required(VERSION 3.21)
project(gemma3_cpp_cli LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 11)

# Prefer Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ------- Options you can tweak --------
# Turn on BLAS (faster CPU GEMM) if available:
set(LLAMA_BLAS ON CACHE BOOL "Enable BLAS for faster CPU inference")
# Choose your vendor (OpenBLAS recommended on Linux; Accelerate on macOS):
# set(LLAMA_BLAS_VENDOR "OpenBLAS" CACHE STRING "BLAS vendor")
# Enforce CPU-only by NOT enabling CUDA/Metal/etc. (defaults are OFF)
# -------------------------------------

include(FetchContent)
FetchContent_Declare(
  llama_cpp
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  # For reproducibility you may pin a commit known to support Gemma 3; master is shown in their README.
  GIT_TAG master
)
FetchContent_MakeAvailable(llama_cpp)

# Our CLI
add_executable(gemma3-cpp src/main.cpp)
target_link_libraries(gemma3-cpp PRIVATE llama)

# On Unix-y systems, link pthread if needed
if(UNIX AND NOT APPLE)
    find_package(Threads REQUIRED)
    target_link_libraries(gemma3-cpp PRIVATE Threads::Threads)
endif()

# Tests (smoke)
enable_testing()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME smoke
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.py $<TARGET_FILE:gemma3-cpp>
)
