# Validation Module CMake Configuration
# This file defines the build configuration for the validation functionality modules

# Define validation module source files
set(VALIDATION_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-validation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test-data-validator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test-data-validator-common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test-data-validator-core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test-data-validator-gguf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test-data-validator-parquet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test-data-validator-text.cpp
)

# Define validation module header files
set(VALIDATION_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/llama-dataset-validation.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test-data-validator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test-data-validator-common.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test-data-validator-core.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test-data-validator-gguf.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test-data-validator-parquet.h
    ${CMAKE_CURRENT_SOURCE_DIR}/test-data-validator-text.h
)

# Create validation library
add_library(dataset_validation STATIC ${VALIDATION_SOURCES} ${VALIDATION_HEADERS})

# Set C++ standard
target_compile_features(dataset_validation PRIVATE cxx_std_17)

# Include directories for validation modules
target_include_directories(dataset_validation PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../core
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
)

# Link core dependencies
target_link_libraries(dataset_validation PRIVATE
    llama
    ggml
)

# Set target properties
set_target_properties(dataset_validation PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Export variables for parent CMakeLists.txt
set(VALIDATION_LIBRARY dataset_validation PARENT_SCOPE)