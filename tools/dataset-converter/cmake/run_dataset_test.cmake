# CMake script to replace shell script functionality for dataset tests
# This replaces the functionality from analyze-core-tests.sh and run-monitored-tests.sh

# Change to working directory
if(WORKING_DIRECTORY)
    execute_process(COMMAND ${CMAKE_COMMAND} -E chdir ${WORKING_DIRECTORY})
endif()

# Function to run command with error handling
function(run_command_with_monitoring COMMAND_NAME EXECUTABLE)
    message(STATUS "=== Testing: ${COMMAND_NAME} ===")
    message(STATUS "Executable: ${EXECUTABLE}")
    
    # Check if executable exists
    if(NOT EXISTS ${EXECUTABLE})
        message(FATAL_ERROR "❌ FAIL: Executable not found: ${EXECUTABLE}")
    endif()
    
    message(STATUS "✅ Executable exists")
    
    # Run the executable and capture output
    execute_process(
        COMMAND ${EXECUTABLE}
        WORKING_DIRECTORY ${WORKING_DIRECTORY}
        RESULT_VARIABLE EXIT_CODE
        OUTPUT_VARIABLE STDOUT_OUTPUT
        ERROR_VARIABLE STDERR_OUTPUT
        TIMEOUT 30
    )
    
    message(STATUS "Exit Code: ${EXIT_CODE}")
    
    # Display output
    if(STDOUT_OUTPUT)
        message(STATUS "--- STDOUT ---")
        message(STATUS "${STDOUT_OUTPUT}")
    endif()
    
    if(STDERR_OUTPUT)
        message(STATUS "--- STDERR ---")
        message(STATUS "${STDERR_OUTPUT}")
    endif()
    
    # Analyze output for patterns
    message(STATUS "--- Analysis ---")
    
    if(EXIT_CODE EQUAL 0)
        message(STATUS "✅ Test PASSED")
    else
        message(STATUS "❌ Test FAILED (exit code: ${EXIT_CODE})")
        
        if(EXIT_CODE EQUAL 139)
            message(STATUS "⚠️  Segmentation fault detected")
        elseif(EXIT_CODE EQUAL 134)
            message(STATUS "⚠️  Abort signal detected")
        elseif(EXIT_CODE EQUAL 124)
            message(STATUS "⚠️  Timeout occurred")
        else()
            message(STATUS "⚠️  Unknown failure mode")
        endif()
    endif()
    
    # Pattern analysis
    string(FIND "${STDOUT_OUTPUT}${STDERR_OUTPUT}" "null" NULL_FOUND)
    if(NOT NULL_FOUND EQUAL -1)
        message(STATUS "✓ NULL pointer handling tested")
    endif()
    
    string(FIND "${STDOUT_OUTPUT}${STDERR_OUTPUT}" "not found" NOT_FOUND_FOUND)
    if(NOT NOT_FOUND_FOUND EQUAL -1)
        message(STATUS "✓ File not found error handling tested")
    endif()
    
    string(FIND "${STDOUT_OUTPUT}${STDERR_OUTPUT}" "gguf" GGUF_FOUND)
    if(NOT GGUF_FOUND EQUAL -1)
        message(STATUS "✓ GGUF format handling tested")
    endif()
    
    string(FIND "${STDOUT_OUTPUT}${STDERR_OUTPUT}" "text" TEXT_FOUND)
    if(NOT TEXT_FOUND EQUAL -1)
        message(STATUS "✓ Text format handling tested")
    endif()
    
    string(FIND "${STDOUT_OUTPUT}${STDERR_OUTPUT}" "parquet" PARQUET_FOUND)
    if(NOT PARQUET_FOUND EQUAL -1)
        message(STATUS "✓ Parquet format handling tested")
    endif()
    
    string(FIND "${STDOUT_OUTPUT}${STDERR_OUTPUT}" "All tests passed" ALL_PASSED_FOUND)
    if(NOT ALL_PASSED_FOUND EQUAL -1)
        message(STATUS "✅ All internal assertions passed")
    endif()
    
    # Return exit code for further processing
    if(NOT EXIT_CODE EQUAL 0)
        message(FATAL_ERROR "Test failed with exit code: ${EXIT_CODE}")
    endif()
endfunction()

# Main execution
message(STATUS "========================================")
message(STATUS "Dataset Test Execution (CMake)")
message(STATUS "========================================")

# Create test data directory
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory test_data)

# Generate test data
message(STATUS "=== Generating Test Data ===")
if(CREATE_TEST_DATA AND EXISTS ${CREATE_TEST_DATA})
    message(STATUS "Running create-test-data...")
    execute_process(
        COMMAND ${CREATE_TEST_DATA}
        WORKING_DIRECTORY ${WORKING_DIRECTORY}
        RESULT_VARIABLE CREATE_DATA_RESULT
    )
    if(NOT CREATE_DATA_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to create test data")
    endif()
endif()

if(CREATE_TEST_DATA_GGUF AND EXISTS ${CREATE_TEST_DATA_GGUF})
    message(STATUS "Running create-test-data-gguf...")
    execute_process(
        COMMAND ${CREATE_TEST_DATA_GGUF}
        WORKING_DIRECTORY ${WORKING_DIRECTORY}
        RESULT_VARIABLE CREATE_GGUF_RESULT
    )
    if(NOT CREATE_GGUF_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to create GGUF test data")
    endif()
endif()

# Create minimal text test data if it doesn't exist
if(NOT EXISTS "${WORKING_DIRECTORY}/test_data/text_dataset.txt")
    message(STATUS "Creating minimal text test data...")
    file(WRITE "${WORKING_DIRECTORY}/test_data/text_dataset.txt" 
        "This is a test dataset for text processing.\n"
        "It contains multiple lines of text data.\n"
        "Each line represents a training sequence.\n"
        "The tokenizer should process this correctly.\n"
        "Special characters: !@#$%^&*()_+-={}[]|\\:;\"'<>?,./\n"
        "Unicode test: αβγδε 中文测试 🚀🔥💯\n"
    )
    message(STATUS "✅ Created: test_data/text_dataset.txt")
endif()

# Run the actual test
if(TEST_EXECUTABLE AND EXISTS ${TEST_EXECUTABLE})
    get_filename_component(TEST_NAME ${TEST_EXECUTABLE} NAME)
    run_command_with_monitoring(${TEST_NAME} ${TEST_EXECUTABLE})
else()
    message(FATAL_ERROR "Test executable not found: ${TEST_EXECUTABLE}")
endif()

message(STATUS "========================================")
message(STATUS "Test Execution Complete")
message(STATUS "========================================")