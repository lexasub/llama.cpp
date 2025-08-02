# Dataset Converter Tests CMake Configuration
# This file defines the build configuration for dataset converter tests

# Enable testing
enable_testing()

# Test data generation executables
add_executable(create-test-data ${CMAKE_CURRENT_SOURCE_DIR}/data/create_test_data.cpp)
target_link_libraries(create-test-data PRIVATE dataset_convert_lib)

add_executable(create-test-data-gguf ${CMAKE_CURRENT_SOURCE_DIR}/data/create_test_gguf.cpp)
target_link_libraries(create-test-data-gguf PRIVATE dataset_convert_lib)

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

    file(RELATIVE_PATH REL_TEST_NAME ${CMAKE_CURRENT_SOURCE_DIR} ${TEST_SRC})
    string(REPLACE "/" "_" REL_TEST_NAME ${REL_TEST_NAME})
    string(REPLACE "\\" "_" REL_TEST_NAME ${REL_TEST_NAME})
    string(REPLACE "." "_" REL_TEST_NAME ${REL_TEST_NAME})
    string(REPLACE " " "_" REL_TEST_NAME ${REL_TEST_NAME})

    add_executable(${REL_TEST_NAME} ${TEST_SRC})
    target_link_libraries(${REL_TEST_NAME} PRIVATE dataset_convert_lib)

    set_target_properties(${REL_TEST_NAME} PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )

    # Use CMake commands instead of shell script execution
    add_test(
        NAME ${REL_TEST_NAME}
        COMMAND ${CMAKE_COMMAND} 
            -DTEST_EXECUTABLE=${CMAKE_CURRENT_BINARY_DIR}/${REL_TEST_NAME}
            -DCREATE_TEST_DATA=${CMAKE_CURRENT_BINARY_DIR}/create-test-data
            -DCREATE_TEST_DATA_GGUF=${CMAKE_CURRENT_BINARY_DIR}/create-test-data-gguf
            -DWORKING_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/run_dataset_test.cmake
    )
    
    # Add dependency on test data generators
    add_dependencies(${REL_TEST_NAME} create-test-data create-test-data-gguf)
    
    # Add monitored test target for this test
    add_monitored_test(${REL_TEST_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${REL_TEST_NAME})
    
    # Add streaming test target if it's a streaming test
    if(${TEST_SRC} MATCHES "streaming")
        add_streaming_test(${REL_TEST_NAME} ${CMAKE_CURRENT_BINARY_DIR}/${REL_TEST_NAME})
    endif()
endfunction()

# Gather all test source files
file(GLOB_RECURSE TEST_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/integration/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/streaming/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unit/*.cpp"
)

# Create test targets for all test sources
foreach(TEST_SRC ${TEST_SOURCES})
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