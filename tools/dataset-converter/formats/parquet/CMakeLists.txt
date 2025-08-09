# Parquet Format Module CMake Configuration
# This file defines the build configuration for the Parquet format handler modules

# Check for Arrow/Parquet dependencies
if(NOT LLAMA_PARQUET)
    message(STATUS "Parquet support disabled - skipping Parquet format modules")
    return()
endif()

# Find required dependencies
find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# Define Parquet module source files
set(PARQUET_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-parquet-core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-parquet-schema.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-parquet-conversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-parquet-streaming.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-parquet-loader.cpp
)

# Define Parquet module header files
set(PARQUET_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-parquet.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-parquet-internal.h
)

# Create Parquet format library
add_library(dataset_parquet_format STATIC ${PARQUET_SOURCES} ${PARQUET_HEADERS})

# Set C++ standard
target_compile_features(dataset_parquet_format PRIVATE cxx_std_17)

# Include directories for Parquet modules
target_include_directories(dataset_parquet_format PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
)

# Link core dependencies and Arrow/Parquet dependencies
target_link_libraries(dataset_parquet_format PRIVATE
    dataset_core
    llama
    ggml
    Arrow::arrow_shared
    Parquet::parquet_shared
)

# Add compile definitions
target_compile_definitions(dataset_parquet_format PRIVATE
    LLAMA_PARQUET
)

# Set target properties
set_target_properties(dataset_parquet_format PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Export variables for parent CMakeLists.txt
set(PARQUET_FORMAT_LIBRARY dataset_parquet_format PARENT_SCOPE)
