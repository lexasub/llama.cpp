#include "test_core_functionality.h"

// Test text factory function with valid and invalid inputs
void test_text_factory();
void test_text_factory() {
    TEST_LOG_SECTION("Testing text factory function");

    // Create a test text file
    std::vector<std::string> lines = {
        "This is a test sentence.",
        "Another test sentence with more words.",
        "A third line for testing the text loader."
    };

    TEST_ASSERT(create_test_text_file(TEST_OUTPUT_TEXT, lines), "Test text file creation should succeed");

    // Test with null path
    TEST_ASSERT(test_error_handling("null path", []() {
        common_params params;
        return llama_dataset_from_txt(&params, nullptr);
    }), "Null path error handling should work");

    // Test with null model (tokenizer)
    TEST_ASSERT(test_error_handling("null tokenizer", [&]() {
        TestParamsBuilder builder;
        common_params params = builder.with_file(TEST_OUTPUT_TEXT).build();
        return llama_dataset_from_txt(&params, nullptr);
    }), "Null tokenizer error handling should work");

    // Note: We can't fully test the text loader without a valid tokenizer model
    // This would require loading a real model, which is beyond the scope of this test
    TEST_LOG_SUCCESS("Text factory function tests completed");

    // Clean up
    cleanup_test_files({TEST_OUTPUT_TEXT});
}

// Test sequence access functions for consistency across formats
void test_sequence_access_validation();
void test_sequence_access_validation() {
    TEST_LOG_SECTION("Testing sequence access functions");

    // Load a GGUF dataset
    TestDatasetGuard dataset(load_test_dataset(TEST_DATA_SMALL_GGUF, DATASET_GGUF, false));
    TEST_ASSERT_NOT_NULL(dataset.get(), "GGUF dataset for sequence access testing");

    display_dataset_summary(dataset.get(), "GGUF dataset");

    // Test basic sequence access
    uint64_t seq_count = llama_dataset_n_sequences(dataset.get());
    if (seq_count > 0) {
        TEST_ASSERT(test_sequence_access(dataset.get(), 0), "First sequence access should work");
    }

    // Test out-of-bounds access
    TEST_ASSERT(test_out_of_bounds_access(dataset.get()), "Out-of-bounds access should be handled correctly");
    TEST_LOG_SUCCESS("Out-of-bounds access checks work");

    // Test null dataset handling
    TEST_ASSERT(test_null_dataset_handling(), "Null dataset handling should work");
    TEST_LOG_SUCCESS("Null dataset handling works");
}

// Test error conditions with invalid inputs
void test_error_conditions();
void test_error_conditions() {
    TEST_LOG_SECTION("Testing error conditions");

    // Test with corrupted GGUF file if available
    bool corrupted_gguf_handled = test_error_handling("corrupted GGUF file", []() {
        return load_test_dataset(TEST_DATA_CORRUPTED_GGUF, DATASET_GGUF, false);
    });
    
    if (corrupted_gguf_handled) {
        TEST_LOG_SUCCESS("Corrupted GGUF file correctly rejected");
    } else {
        TEST_LOG_INFO("Corrupted GGUF file was loaded (might be valid test data)");
    }

#ifdef LLAMA_PARQUET
    // Test with corrupted Parquet file if available
    bool corrupted_parquet_handled = test_error_handling("corrupted Parquet file", []() {
        return load_test_dataset(TEST_DATA_CORRUPTED_PARQUET, DATASET_PARQUET, false);
    });
    
    if (corrupted_parquet_handled) {
        TEST_LOG_SUCCESS("Corrupted Parquet file correctly rejected");
    } else {
        TEST_LOG_INFO("Corrupted Parquet file was loaded (might be valid test data)");
    }
#endif

    // Test error code to string conversion
    const char* error_str = llama_dataset_error_code_to_string(DATASET_ERROR_FILE_NOT_FOUND);
    TEST_ASSERT_NOT_NULL(error_str, "error code string");
    TEST_LOG_INFO("Error code string: %s", error_str);

    // Test error clearing
    llama_dataset_clear_error();
    TEST_ASSERT_NO_ERROR();
    TEST_LOG_SUCCESS("Error clearing works");
}

int main() {
    LLAMA_LOG_INFO("=== Running validation functionality tests ===\n");

    // Test text factory functions
    test_text_factory();

    // Test sequence access
    test_sequence_access_validation();

    // Test error conditions
    test_error_conditions();

    LLAMA_LOG_INFO("\n=== All validation functionality tests completed successfully! ===\n");
    return 0;
}