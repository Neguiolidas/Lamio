cmake_minimum_required(VERSION 3.20)
project(lamio_v2 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Lamio v2 links the ggml library from the surrounding llama.cpp-like build.
# The ggml headers live at ggml/include, the built library at
# ${GGML_BUILD_DIR}/bin/libggml.so.
set(GGML_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../ggml CACHE PATH "Path to ggml root")
set(GGML_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../build CACHE PATH "Path to built ggml binaries")

add_subdirectory(src)

# If ggml static/shared libs are not found in the build tree, fail loudly.
find_library(GGML_LIB NAMES ggml PATHS ${GGML_BUILD_DIR}/bin NO_DEFAULT_PATH)
if(NOT GGML_LIB)
  message(FATAL_ERROR "libggml not found in ${GGML_BUILD_DIR}/bin. Run the llama.cpp/fork build first.")
endif()
message(STATUS "Lamio v2 linking libggml: ${GGML_LIB}")