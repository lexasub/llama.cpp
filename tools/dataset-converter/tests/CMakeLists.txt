# Dataset Converter Tests CMake Configuration
# This file defines the build configuration for dataset converter tests

# Enable testing
enable_testing()

# Ensure all new test modules are properly discovered
# New test modules created during refactoring:
# - test_gguf_functionality.cpp (GGUF format tests)
# - test_parquet_functionality.cpp (Parquet format tests) 
# - test_streaming_functionality.cpp (Streaming tests)
# - test_validation_functionality.cpp (Validation tests)
message(STATUS "Configuring dataset converter test modules...")

# Test data generation executables
add_executable(create-test-data ${CMAKE_CURRENT_SOURCE_DIR}/data/create_test_data.cpp)
target_link_libraries(create-test-data PRIVATE dataset_convert_lib common llama ggml)

add_executable(create-test-data-gguf ${CMAKE_CURRENT_SOURCE_DIR}/data/create_test_gguf.cpp)
target_link_libraries(create-test-data-gguf PRIVATE dataset_convert_lib common llama ggml)

# Set output directory for test data generators
set_target_properties(create-test-data create-test-data-gguf PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Custom target for test data generation
add_custom_target(generate-test-data
    COMMAND ${CMAKE_COMMAND} -E echo "Generating test data..."
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/create-test-data
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/create-test-data-gguf
    COMMAND ${CMAKE_COMMAND} -E echo "Test data generation completed"
    DEPENDS create-test-data create-test-data-gguf
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating test data for dataset converter"
)

# Function to create monitored test execution
function(add_monitored_test TEST_NAME TEST_EXECUTABLE)
    add_custom_target(monitor-${TEST_NAME}
        COMMAND ${CMAKE_COMMAND} -E echo "========================================"
        COMMAND ${CMAKE_COMMAND} -E echo "Core Dataset Functionality Test Analysis"
        COMMAND ${CMAKE_COMMAND} -E echo "========================================"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "=== Testing: ${TEST_NAME} ==="
        COMMAND ${CMAKE_COMMAND} -E echo "Executable: ${TEST_EXECUTABLE}"
        COMMAND ${CMAKE_COMMAND} -E echo "✅ Executable exists and is runnable"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "--- Execution Output ---"
        COMMAND ${CMAKE_COMMAND} -E time ${TEST_EXECUTABLE}
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "✅ Test execution completed"
        DEPENDS ${TEST_EXECUTABLE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running monitored test for ${TEST_NAME}"
    )
endfunction()

# Function to create streaming test execution
function(add_streaming_test TEST_NAME TEST_EXECUTABLE)
    add_custom_target(streaming-${TEST_NAME}
        COMMAND ${CMAKE_COMMAND} -E echo "========================================"
        COMMAND ${CMAKE_COMMAND} -E echo "Streaming Functionality Test Analysis"
        COMMAND ${CMAKE_COMMAND} -E echo "========================================"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "=== Testing: ${TEST_NAME} ==="
        COMMAND ${CMAKE_COMMAND} -E echo "Executable: ${TEST_EXECUTABLE}"
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "--- Execution with Memory Monitoring ---"
        COMMAND ${CMAKE_COMMAND} -E time ${TEST_EXECUTABLE}
        COMMAND ${CMAKE_COMMAND} -E echo ""
        COMMAND ${CMAKE_COMMAND} -E echo "✅ Streaming test execution completed"
        DEPENDS ${TEST_EXECUTABLE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running streaming test for ${TEST_NAME}"
    )
endfunction()

# Enhanced test function that uses CMake commands instead of shell scripts
function(add_test_target TEST_SRC)
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

    # Create clearer test names using directory prefixes
    file(RELATIVE_PATH REL_TEST_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${TEST_SRC})
    get_filename_component(TEST_DIR ${REL_TEST_PATH} DIRECTORY)

    # Generate clear test target name with dataset prefix and directory
    if(TEST_DIR MATCHES "unit")
        set(TARGET_NAME "dataset_unit_${TEST_NAME}")
    elseif(TEST_DIR MATCHES "integration")
        set(TARGET_NAME "dataset_integration_${TEST_NAME}")
    elseif(TEST_DIR MATCHES "streaming")
        set(TARGET_NAME "dataset_streaming_${TEST_NAME}")
    else()
        set(TARGET_NAME "dataset_${TEST_NAME}")
    endif()

    # Check if this test needs utilities (functionality tests or integration tests)
    if(${TEST_SRC} MATCHES "test_.*_functionality\\.cpp" OR ${TEST_SRC} MATCHES "integration" OR ${TEST_SRC} MATCHES "streaming")
        # Add the utilities source file for tests that need shared utilities
        add_executable(${TARGET_NAME} ${TEST_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/unit/test_core_functionality_utils.cpp)
    else()
        add_executable(${TARGET_NAME} ${TEST_SRC})
    endif()
    
    target_link_libraries(${TARGET_NAME} PRIVATE dataset_convert_lib common llama ggml)
    
    # Ensure proper linking for new test modules
    if(${TEST_SRC} MATCHES "test_.*_functionality\\.cpp")
        # Additional linking for functionality test modules
        target_include_directories(${TARGET_NAME} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/../core
            ${CMAKE_CURRENT_SOURCE_DIR}/../formats
            ${CMAKE_CURRENT_SOURCE_DIR}/../streaming
            ${CMAKE_CURRENT_SOURCE_DIR}/../validation
        )
        
        # Link format-specific libraries based on test type
        if(${TEST_SRC} MATCHES "parquet" AND LLAMA_PARQUET)
            target_link_libraries(${TARGET_NAME} PRIVATE dataset_parquet_format)
        endif()
    endif()

    set_target_properties(${TARGET_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # Use CMake commands instead of shell script execution
    add_test(
        NAME ${TARGET_NAME}
        COMMAND ${CMAKE_COMMAND}
            -DTEST_EXECUTABLE=${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}
            -DCREATE_TEST_DATA=${CMAKE_CURRENT_BINARY_DIR}/create-test-data
            -DCREATE_TEST_DATA_GGUF=${CMAKE_CURRENT_BINARY_DIR}/create-test-data-gguf
            -DWORKING_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/run_dataset_test.cmake
    )

    # Add dependency on test data generators
    add_dependencies(${TARGET_NAME} create-test-data create-test-data-gguf)

    # Add monitored test target for this test
    add_monitored_test(${TARGET_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME})

    # Add streaming test target if it's a streaming test
    if(${TEST_SRC} MATCHES "streaming")
        add_streaming_test(${TARGET_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME})
    endif()
endfunction()

# Gather all test source files with conditional Parquet support
file(GLOB_RECURSE UNIT_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp")
file(GLOB_RECURSE INTEGRATION_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/integration/*.cpp")
file(GLOB_RECURSE STREAMING_TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/streaming/*.cpp")

# Combine all test sources
set(TEST_SOURCES ${UNIT_TEST_SOURCES} ${INTEGRATION_TEST_SOURCES} ${STREAMING_TEST_SOURCES})

# Filter out Parquet-related tests if LLAMA_PARQUET is not enabled
if(NOT LLAMA_PARQUET)
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*parquet.*")
    list(FILTER TEST_SOURCES EXCLUDE REGEX ".*arrow.*")
    message(STATUS "LLAMA_PARQUET is disabled, excluding Parquet-related tests")
else()
    message(STATUS "LLAMA_PARQUET is enabled, including all tests")
endif()

# Create test targets for all test sources
foreach(TEST_SRC ${TEST_SOURCES})
    # Skip utility files that don't contain main functions
    if(${TEST_SRC} MATCHES ".*_utils\\.cpp$")
        continue()
    endif()
    
    # Skip Parquet tests if support is not available
    if(NOT LLAMA_PARQUET AND (${TEST_SRC} MATCHES "parquet" OR ${TEST_SRC} MATCHES "arrow"))
        continue()
    endif()

    add_test_target(${TEST_SRC})
endforeach()

# Custom targets for comprehensive testing
add_custom_target(dataset-test-analysis
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/dataset_analysis.cmake
    DEPENDS generate-test-data
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running comprehensive dataset test analysis"
)

add_custom_target(analyze-core-tests
    COMMAND ${CMAKE_COMMAND}
        -DWORKING_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
        -DTEST_EXECUTABLES="${CMAKE_CURRENT_BINARY_DIR}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/analyze_core_tests.cmake
    DEPENDS generate-test-data
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Analyzing core dataset functionality tests"
)

add_custom_target(run-monitored-tests
    COMMAND ${CMAKE_COMMAND}
        -DWORKING_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
        -DTEST_EXECUTABLES="${CMAKE_CURRENT_BINARY_DIR}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/run_monitored_tests.cmake
    DEPENDS generate-test-data
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running monitored streaming functionality tests"
)

add_custom_target(setup-dataset-environment
    COMMAND ${CMAKE_COMMAND} -E echo "Setting up dataset converter environment..."
    COMMAND ${CMAKE_COMMAND} -E make_directory test_data
    COMMAND ${CMAKE_COMMAND} -E make_directory test_output
    COMMAND ${CMAKE_COMMAND} -E make_directory temp
    COMMAND ${CMAKE_COMMAND} -E echo "Environment setup completed"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Setting up dataset converter environment"
)

add_custom_target(generate-datasets
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/generate_datasets.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating comprehensive test datasets"
)

add_custom_target(run-all-dataset-tests
    COMMAND ${CMAKE_COMMAND} -E echo "Running comprehensive dataset converter tests..."
    DEPENDS analyze-core-tests run-monitored-tests dataset-test-analysis generate-datasets
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all dataset converter tests"
)

# Verification target to ensure all new test modules build correctly
add_custom_target(verify-test-modules
    COMMAND ${CMAKE_COMMAND} -E echo "Verifying all test modules build correctly..."
    COMMAND ${CMAKE_COMMAND} -E echo "✅ test_gguf_functionality.cpp"
    COMMAND ${CMAKE_COMMAND} -E echo "✅ test_parquet_functionality.cpp" 
    COMMAND ${CMAKE_COMMAND} -E echo "✅ test_streaming_functionality.cpp"
    COMMAND ${CMAKE_COMMAND} -E echo "✅ test_validation_functionality.cpp"
    COMMAND ${CMAKE_COMMAND} -E echo "All new test modules configured successfully"
    DEPENDS dataset_unit_test_gguf_functionality dataset_unit_test_parquet_functionality dataset_unit_test_streaming_functionality dataset_unit_test_validation_functionality
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Verifying new test module configuration"
)
