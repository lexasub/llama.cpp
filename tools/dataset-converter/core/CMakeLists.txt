# Core Dataset Module CMake Configuration
# This file defines the build configuration for the core dataset functionality

# Define core module source files (after reorganization - streaming and loaders moved to separate modules)
set(CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-error.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-metadata.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-conversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-sequence.cpp

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
)

# Create core dataset library
add_library(dataset_core STATIC ${CORE_SOURCES} ${CORE_HEADERS})

# Set C++ standard
target_compile_features(dataset_core PRIVATE cxx_std_17)

# Include directories for core modules (includes streaming for internal functions)
target_include_directories(dataset_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../streaming
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)

# Public include directories for other modules to use
target_include_directories(dataset_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Find threading support if not already found
if(NOT TARGET Threads::Threads)
    find_package(Threads REQUIRED)
endif()

# Link core dependencies and cross-module dependencies
target_link_libraries(dataset_core PRIVATE
    llama
    ggml
    Threads::Threads
)

# Link with streaming and format modules (will be linked by parent CMakeLists.txt)
# This ensures proper cross-module linking after reorganization

# Set target properties
set_target_properties(dataset_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Export variables for parent CMakeLists.txt
set(CORE_LIBRARY dataset_core PARENT_SCOPE)