add_library(lamio_core STATIC
    gguf_reader.cpp
    model_config.cpp
    block_tensors.cpp
    tokenizer.cpp
)

target_include_directories(lamio_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GGML_ROOT}/include
)

# ggml.c separated into ggml.so (base ops) + ggml-base.so (GGUF parser, etc.).
find_library(GGML_BASE_LIB NAMES ggml-base PATHS ${GGML_BUILD_DIR}/bin NO_DEFAULT_PATH)
if(GGML_BASE_LIB)
  message(STATUS "Lamio v2 also linking libggml-base: ${GGML_BASE_LIB}")
endif()

target_link_libraries(lamio_core PUBLIC ${GGML_LIB} ${GGML_BASE_LIB})
target_compile_options(lamio_core PRIVATE -Wall -Wextra -O3)

add_executable(lamio
    main.cpp
)

target_link_libraries(lamio PRIVATE lamio_core)