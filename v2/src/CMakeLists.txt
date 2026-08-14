add_library(lamio_core STATIC
    gguf_reader.cpp
    model_config.cpp
    block_tensors.cpp
    tokenizer.cpp
    weight_loader.cpp
    dense_forward.cpp
    qwen35_forward.cpp
)

target_include_directories(lamio_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GGML_ROOT}/include
)

# ggml.c separated into ggml.so (base ops) + ggml-base.so (GGUF parser, etc.).
# ggml libs: base (GGUF parser), cpu (CPU backend), cuda (GPU backend, loaded via dlopen at runtime)
find_library(GGML_BASE_LIB NAMES ggml-base PATHS ${GGML_BUILD_DIR}/bin NO_DEFAULT_PATH)
find_library(GGML_CPU_LIB  NAMES ggml-cpu  PATHS ${GGML_BUILD_DIR}/bin NO_DEFAULT_PATH)
find_library(GGML_CUDA_LIB NAMES ggml-cuda PATHS ${GGML_BUILD_DIR}/bin NO_DEFAULT_PATH)
if(GGML_BASE_LIB)
  message(STATUS "Lamio v2 linking libggml-base: ${GGML_BASE_LIB}")
endif()
if(GGML_CPU_LIB)
  message(STATUS "Lamio v2 linking libggml-cpu: ${GGML_CPU_LIB}")
endif()
if(GGML_CUDA_LIB)
  message(STATUS "Lamio v2 linking libggml-cuda: ${GGML_CUDA_LIB}")
  target_link_libraries(lamio_core PUBLIC ${GGML_CUDA_LIB} dl)
endif()

target_link_libraries(lamio_core PUBLIC ${GGML_LIB} ${GGML_BASE_LIB} ${GGML_CPU_LIB} pthread)
target_compile_options(lamio_core PRIVATE -Wall -Wextra -O3)

add_executable(lamio
    main.cpp
)

target_link_libraries(lamio PRIVATE lamio_core)