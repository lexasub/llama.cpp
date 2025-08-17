cmake_minimum_required(VERSION 3.14)
project(llama-dataset-converter)

# Dataset Converter CMake Configuration

# Define standardized include directories for all dataset-converter modules
set(DATASET_CONVERTER_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/formats
    ${CMAKE_CURRENT_SOURCE_DIR}/streaming
    ${CMAKE_CURRENT_SOURCE_DIR}/tokenization
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks
    ${CMAKE_CURRENT_SOURCE_DIR}/performance
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/validation
    ${CMAKE_CURRENT_SOURCE_DIR}/../../common
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ggml/include
)

# Function to add standardized include directories to dataset-converter targets
function(add_dataset_converter_includes target_name)
    target_include_directories(${target_name} PRIVATE ${DATASET_CONVERTER_INCLUDE_DIRS})
endfunction()

# Function to add standardized include directories with public visibility
function(add_dataset_converter_includes_public target_name)
    get_target_property(target_type ${target_name} TYPE)
    if(target_type STREQUAL "INTERFACE_LIBRARY")
        target_include_directories(${target_name} INTERFACE ${DATASET_CONVERTER_INCLUDE_DIRS})
    else()
        target_include_directories(${target_name} PUBLIC ${DATASET_CONVERTER_INCLUDE_DIRS})
    endif()
endfunction()

# Make the include directories available to all subdirectories
set(DATASET_CONVERTER_INCLUDE_DIRS ${DATASET_CONVERTER_INCLUDE_DIRS} CACHE INTERNAL "Dataset converter include directories")

# Add core subdirectory first to ensure it's available for other modules
add_subdirectory(core)

# Add format subdirectories
add_subdirectory(formats/gguf)
add_subdirectory(formats/text)

# Add platform subdirectory
add_subdirectory(platform)

# Add validation subdirectory
add_subdirectory(validation)

# Add streaming subdirectory
add_subdirectory(streaming)

# Add tokenization subdirectory
add_subdirectory(tokenization)

# Add formats subdirectory
add_subdirectory(formats)

# Conditionally add parquet format if enabled
if(LLAMA_PARQUET)
    add_subdirectory(formats/parquet)
endif()

# Add test-utils subdirectory
add_subdirectory(test-utils)

# Add tests subdirectory
add_subdirectory(tests)

# Add benchmarks subdirectory
add_subdirectory(benchmarks)

# Add performance framework subdirectory
add_subdirectory(performance)

# Add dataset converter core library (shared library combining all modules)
add_library(llama-dataset-core SHARED
    llama-dataset-core-wrapper.cpp
)

# Apply standardized include directories
add_dataset_converter_includes_public(llama-dataset-core)

# Link all the static libraries into the shared library
target_link_libraries(llama-dataset-core PRIVATE
    -Wl,--whole-archive
    dataset_core
    dataset_platform
    dataset_streaming
    dataset_gguf_format
    dataset_text_format
    # dataset_validation  # Temporarily disabled due to interface issues
    -Wl,--no-whole-archive
)

# Link external libraries as public so they're available to consumers
target_link_libraries(llama-dataset-core PUBLIC
    llama
    ggml
)

# Configure proper RPATH and library properties
set_target_properties(llama-dataset-core PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH_USE_LINK_PATH ON
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

# Conditionally link parquet format if enabled
if(LLAMA_PARQUET)
    target_link_libraries(llama-dataset-core PUBLIC dataset_parquet_format)
endif()

# Removed obsolete G5 test executables that were never implemented

# Add validation test executable
add_executable(test-validation-enhanced
    tests/test_validation_enhanced.cpp
)

# Apply standardized include directories
add_dataset_converter_includes(test-validation-enhanced)

target_link_libraries(test-validation-enhanced PRIVATE
    llama-dataset-core
    llama
    ggml
    gtest
    gtest_main
    gmock
)

# Add validation command-line tool
add_executable(llama-dataset-validate
    tools/llama-dataset-validate.cpp
)

# Apply standardized include directories
add_dataset_converter_includes(llama-dataset-validate)

target_link_libraries(llama-dataset-validate PRIVATE
    llama-dataset-core
    llama
    ggml
)

# Add I3 enhanced validation test
add_executable(test-i3-enhanced-validation
    tests/test_enhanced_validation.cpp
)

# Apply standardized include directories
add_dataset_converter_includes(test-i3-enhanced-validation)

target_link_libraries(test-i3-enhanced-validation PRIVATE
    llama-dataset-core
    llama
    ggml
)

# Add I3 enhanced validation demo
add_executable(demo-i3-enhanced-validation
    tests/demo_enhanced_validation.cpp
)

# Apply standardized include directories
add_dataset_converter_includes(demo-i3-enhanced-validation)

target_link_libraries(demo-i3-enhanced-validation PRIVATE
    llama-dataset-core
    llama
    ggml
)

# Task H3: Enhanced testing and validation infrastructure
add_executable(test-h3-build-validation
    tests/test_h3_build_validation.cpp
)

target_compile_features(test-h3-build-validation PRIVATE cxx_std_17)

# Find threading support
find_package(Threads REQUIRED)
target_link_libraries(test-h3-build-validation PRIVATE Threads::Threads)

# Enhanced infrastructure test with command-line support
add_executable(test-h3-enhanced-infrastructure
    tests/test_h3_enhanced_infrastructure.cpp
)

target_compile_features(test-h3-enhanced-infrastructure PRIVATE cxx_std_17)
target_link_libraries(test-h3-enhanced-infrastructure PRIVATE Threads::Threads)

# Add tests to CTest
enable_testing()
add_test(NAME h3-build-validation COMMAND test-h3-build-validation)
set_tests_properties(h3-build-validation PROPERTIES
    TIMEOUT 30
    LABELS "h3;build;validation;infrastructure"
)

add_test(NAME h3-enhanced-infrastructure COMMAND test-h3-enhanced-infrastructure --verbose --generate-report)
set_tests_properties(h3-enhanced-infrastructure PROPERTIES
    TIMEOUT 60
    LABELS "h3;enhanced;infrastructure;performance"
)

# Test with dataset path argument
add_test(NAME h3-enhanced-infrastructure-with-dataset 
         COMMAND test-h3-enhanced-infrastructure --dataset-path ./test_data --verbose --generate-report --report-path h3_dataset_test_report.txt)
set_tests_properties(h3-enhanced-infrastructure-with-dataset PROPERTIES
    TIMEOUT 60
    LABELS "h3;enhanced;infrastructure;dataset;performance"
)

# Removed obsolete streaming cache integration test

# Task I4: Registry performance cache optimization test
add_executable(test-registry-cache-performance
    tests/test_registry_cache_performance.cpp
)

# Apply standardized include directories
add_dataset_converter_includes(test-registry-cache-performance)

target_link_libraries(test-registry-cache-performance PRIVATE
    llama-dataset-core
    dataset_core
    llama
    ggml
)

# Add test to CTest
add_test(NAME registry-cache-performance COMMAND test-registry-cache-performance)
set_tests_properties(registry-cache-performance PROPERTIES
    TIMEOUT 60
    LABELS "i4;registry;cache;performance;optimization"
)