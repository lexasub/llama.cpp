# GGUF Format Module CMake Configuration
# This file defines the build configuration for the GGUF format handler modules

# Define GGUF module source files
set(GGUF_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-gguf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-gguf-utils.cpp
)

# Define GGUF module header files
set(GGUF_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-gguf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-gguf-utils.h
)

# Create GGUF format library
add_library(dataset_gguf_format STATIC ${GGUF_SOURCES} ${GGUF_HEADERS})

# Set C++ standard
target_compile_features(dataset_gguf_format PRIVATE cxx_std_17)

# Include directories for GGUF modules
target_include_directories(dataset_gguf_format PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
)

# Link core dependencies
target_link_libraries(dataset_gguf_format PRIVATE
    llama
    ggml
)

# Set target properties
set_target_properties(dataset_gguf_format PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Export variables for parent CMakeLists.txt
set(GGUF_FORMAT_LIBRARY dataset_gguf_format PARENT_SCOPE)