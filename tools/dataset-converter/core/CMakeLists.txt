# Core Dataset Module CMake Configuration
# This file defines the build configuration for the core dataset functionality

# Define core module source files
set(CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-sequence.cpp
)

# Define core module header files
set(CORE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-internal.h
)

# Create core dataset library
add_library(dataset_core STATIC ${CORE_SOURCES} ${CORE_HEADERS})

# Set C++ standard
target_compile_features(dataset_core PRIVATE cxx_std_17)

# Include directories for core modules
target_include_directories(dataset_core PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)

# Link core dependencies
target_link_libraries(dataset_core PRIVATE
    llama
    ggml
    Threads::Threads
)

# Set target properties
set_target_properties(dataset_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Export variables for parent CMakeLists.txt
set(CORE_LIBRARY dataset_core PARENT_SCOPE)