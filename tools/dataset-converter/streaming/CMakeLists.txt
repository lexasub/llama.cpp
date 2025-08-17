# Streaming Module CMake Configuration
# This file defines the build configuration for the streaming functionality modules

# Define streaming module source files
set(STREAMING_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-streaming.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/streaming-cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/streaming-cache-dod.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/streaming-memory-monitor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/streaming-optimization-manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/streaming-read-ahead.cpp
)

# Define streaming module header files
set(STREAMING_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/streaming-cache.h
    ${CMAKE_CURRENT_SOURCE_DIR}/streaming-cache-dod.h
    ${CMAKE_CURRENT_SOURCE_DIR}/streaming-memory-monitor.h
    ${CMAKE_CURRENT_SOURCE_DIR}/streaming-optimization-manager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/streaming-read-ahead.h
)

# Create streaming library
add_library(dataset_streaming STATIC ${STREAMING_SOURCES} ${STREAMING_HEADERS})

# Set C++ standard
target_compile_features(dataset_streaming PRIVATE cxx_std_17)

# Include directories for streaming modules using standardized paths
if(DEFINED DATASET_CONVERTER_INCLUDE_DIRS)
    target_include_directories(dataset_streaming PRIVATE ${DATASET_CONVERTER_INCLUDE_DIRS})
else()
    # Fallback to manual include directories if parent hasn't defined them
    target_include_directories(dataset_streaming PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../core
        ${CMAKE_CURRENT_SOURCE_DIR}/../platform
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../common
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../src
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )
endif()

# Find threading support if not already found
if(NOT TARGET Threads::Threads)
    find_package(Threads REQUIRED)
endif()

# Link core dependencies and cross-module dependencies
target_link_libraries(dataset_streaming PUBLIC
    dataset_platform
    Threads::Threads
)

# Set target properties
set_target_properties(dataset_streaming PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Create alias for compatibility
add_library(streaming_cache ALIAS dataset_streaming)

# Export variables for parent CMakeLists.txt
set(STREAMING_LIBRARY dataset_streaming PARENT_SCOPE)