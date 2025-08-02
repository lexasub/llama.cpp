# Text Format Module CMake Configuration
# This file defines the build configuration for the text format handler modules

# Define text module source files
set(TEXT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-text.cpp
)

# Define text module header files
set(TEXT_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-text.h
)

# Create text format library
add_library(dataset_text_format STATIC ${TEXT_SOURCES} ${TEXT_HEADERS})

# Set C++ standard
target_compile_features(dataset_text_format PRIVATE cxx_std_17)

# Include directories for text modules
target_include_directories(dataset_text_format PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include
)

# Link core dependencies
target_link_libraries(dataset_text_format PRIVATE
    llama
    ggml
)

# Set target properties
set_target_properties(dataset_text_format PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Export variables for parent CMakeLists.txt
set(TEXT_FORMAT_LIBRARY dataset_text_format PARENT_SCOPE)