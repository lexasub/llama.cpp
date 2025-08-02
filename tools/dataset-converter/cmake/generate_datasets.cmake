# CMake script for dataset generation
# This replaces any shell script functionality for creating test datasets

message(STATUS "========================================")
message(STATUS "Dataset Generation Script")
message(STATUS "========================================")

# Function to create test datasets
function(create_test_datasets)
    message(STATUS "Creating comprehensive test datasets...")
    
    # Create directories
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory test_data)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory test_data/gguf)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory test_data/text)
    execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory test_data/parquet)
    
    # Create text datasets
    message(STATUS "Creating text datasets...")
    
    # Basic text dataset
    file(WRITE "test_data/text/basic_dataset.txt"
        "This is a basic text dataset for testing.\n"
        "It contains simple sentences for tokenization.\n"
        "Each line represents a training example.\n"
        "The dataset should be processed correctly by the converter.\n"
    )
    
    # Large text dataset
    set(LARGE_TEXT_CONTENT "")
    foreach(i RANGE 1 100)
        string(APPEND LARGE_TEXT_CONTENT "Line ${i}: This is a longer text dataset with more content for testing streaming functionality.\n")
    endforeach()
    file(WRITE "test_data/text/large_dataset.txt" ${LARGE_TEXT_CONTENT})
    
    # Unicode text dataset
    file(WRITE "test_data/text/unicode_dataset.txt"
        "English: Hello world!\n"
        "Spanish: ¡Hola mundo!\n"
        "French: Bonjour le monde!\n"
        "German: Hallo Welt!\n"
        "Chinese: 你好世界！\n"
        "Japanese: こんにちは世界！\n"
        "Russian: Привет мир!\n"
        "Arabic: مرحبا بالعالم!\n"
        "Emoji: 🌍🚀💯🔥⭐\n"
    )
    
    # Special characters dataset
    file(WRITE "test_data/text/special_chars_dataset.txt"
        "Special characters test:\n"
        "Punctuation: !@#$%^&*()_+-={}[]|\\:;\"'<>?,./\n"
        "Math symbols: ∑∏∫∆∇∂√∞±≤≥≠≈\n"
        "Currency: $€£¥₹₽₩₪\n"
        "Arrows: ←→↑↓↔↕⇐⇒⇑⇓\n"
    )
    
    message(STATUS "✅ Created text datasets")
    
    # Create corrupted test files for error handling
    message(STATUS "Creating corrupted test files...")
    
    file(WRITE "test_data/corrupted_gguf.gguf" "INVALID_GGUF_HEADER\xFF\xFF\xFF\xFF")
    file(WRITE "test_data/empty_file.txt" "")
    file(WRITE "test_data/binary_junk.bin" "\x00\x01\x02\x03\xFF\xFE\xFD\xFC")
    
    message(STATUS "✅ Created corrupted test files")
    
    # Create configuration files
    message(STATUS "Creating configuration files...")
    
    file(WRITE "test_data/converter_config.json"
        "{\n"
        "  \"input_format\": \"text\",\n"
        "  \"output_format\": \"gguf\",\n"
        "  \"streaming_enabled\": true,\n"
        "  \"batch_size\": 1024,\n"
        "  \"memory_limit\": \"1GB\"\n"
        "}\n"
    )
    
    message(STATUS "✅ Created configuration files")
endfunction()

# Function to validate created datasets
function(validate_datasets)
    message(STATUS "")
    message(STATUS "=== Validating Created Datasets ===")
    
    set(EXPECTED_FILES
        "test_data/text/basic_dataset.txt"
        "test_data/text/large_dataset.txt"
        "test_data/text/unicode_dataset.txt"
        "test_data/text/special_chars_dataset.txt"
        "test_data/corrupted_gguf.gguf"
        "test_data/empty_file.txt"
        "test_data/binary_junk.bin"
        "test_data/converter_config.json"
    )
    
    foreach(FILE ${EXPECTED_FILES})
        if(EXISTS ${FILE})
            file(SIZE ${FILE} FILE_SIZE)
            message(STATUS "✅ Validated: ${FILE} (${FILE_SIZE} bytes)")
        else()
            message(STATUS "❌ Missing: ${FILE}")
        endif()
    endforeach()
endfunction()

# Function to create dataset metadata
function(create_dataset_metadata)
    message(STATUS "")
    message(STATUS "Creating dataset metadata...")
    
    # Get current timestamp
    string(TIMESTAMP CURRENT_TIME "%Y-%m-%d %H:%M:%S")
    
    file(WRITE "test_data/dataset_metadata.json"
        "{\n"
        "  \"created_at\": \"${CURRENT_TIME}\",\n"
        "  \"generator\": \"CMake Dataset Generator\",\n"
        "  \"version\": \"1.0\",\n"
        "  \"datasets\": {\n"
        "    \"text\": {\n"
        "      \"basic_dataset.txt\": \"Simple text for basic testing\",\n"
        "      \"large_dataset.txt\": \"Large text for streaming tests\",\n"
        "      \"unicode_dataset.txt\": \"Unicode character testing\",\n"
        "      \"special_chars_dataset.txt\": \"Special character handling\"\n"
        "    },\n"
        "    \"test_files\": {\n"
        "      \"corrupted_gguf.gguf\": \"Corrupted GGUF for error testing\",\n"
        "      \"empty_file.txt\": \"Empty file for edge case testing\",\n"
        "      \"binary_junk.bin\": \"Binary data for format detection\"\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    
    message(STATUS "✅ Created dataset metadata")
endfunction()

# Main execution
create_test_datasets()
validate_datasets()
create_dataset_metadata()

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Dataset Generation Complete")
message(STATUS "========================================")
message(STATUS "All test datasets have been created successfully.")
message(STATUS "Use 'make run-all-dataset-tests' to run comprehensive tests.")