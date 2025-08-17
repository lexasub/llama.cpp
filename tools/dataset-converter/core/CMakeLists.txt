# Core Dataset Module CMake Configuration
# This file defines the build configuration for the core dataset functionality

# Define core module source files (after reorganization - streaming and loaders moved to separate modules)
set(CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-error.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-metadata.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-conversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-sequence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-memory-optimizations.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-registry-stubs.cpp
    # Registry and validation modules temporarily disabled due to interface issues
    # These will be re-enabled after interface compatibility is resolved
    # ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-registry-core.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-registry-cache.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-registry-format.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-registry-auto.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-validation-core.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-validation-format.cpp
    # ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-validation-reporting.cpp
)

# Define core module header files (no streaming or format-specific headers)
set(CORE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-error.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-metadata.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-conversion.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-internal.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-modules.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-optimized-layout.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-registry.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-registry-core.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-registry-cache.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-registry-format.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-registry-auto.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-validation.h
)

# Create core dataset library
add_library(dataset_core STATIC ${CORE_SOURCES} ${CORE_HEADERS})

# Set C++ standard
target_compile_features(dataset_core PRIVATE cxx_std_17)

# Include directories for core modules using standardized paths
# Note: This assumes the parent CMakeLists.txt has defined DATASET_CONVERTER_INCLUDE_DIRS
if(DEFINED DATASET_CONVERTER_INCLUDE_DIRS)
    target_include_directories(dataset_core PRIVATE ${DATASET_CONVERTER_INCLUDE_DIRS})
else()
    # Fallback to manual include directories if parent hasn't defined them
    target_include_directories(dataset_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../streaming
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
    )
endif()

# Public include directories for other modules to use
target_include_directories(dataset_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Find threading support if not already found
if(NOT TARGET Threads::Threads)
    find_package(Threads REQUIRED)
endif()

# Link core dependencies and cross-module dependencies
target_link_libraries(dataset_core PUBLIC
    Threads::Threads
    ggml
    llama
    dataset_streaming
)

# Link with streaming and format modules (will be linked by parent CMakeLists.txt)
# This ensures proper cross-module linking after reorganization

# Set target properties
set_target_properties(dataset_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Add SIMD optimizations for memory optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(dataset_core PRIVATE
            -mavx2
            -mfma
            -msse4.2
        )
        target_compile_definitions(dataset_core PRIVATE
            __AVX2__=1
        )
    endif()
endif()

# Export variables for parent CMakeLists.txt
set(CORE_LIBRARY dataset_core PARENT_SCOPE)