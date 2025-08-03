#include "test_core_functionality.h"

// Test GGUF factory function with valid and invalid inputs
void test_gguf_factory();
void test_gguf_factory() {
    TEST_LOG_SECTION("Testing GGUF factory function");

    // Test with null path
    TEST_ASSERT(test_error_handling("null path", []() {
        common_params params;
        return llama_dataset_from_gguf(&params);
    }), "Null path error handling should work");

    // Test with non-existent file
    TEST_ASSERT(test_error_handling("non-existent file", []() {
        TestParamsBuilder builder;
        common_params params = builder.with_file("non_existent_file.gguf").build();
        return llama_dataset_from_gguf(&params);
    }), "Non-existent file error handling should work");

    // Test with valid file
    TestDatasetGuard dataset(load_test_dataset(TEST_DATA_SMALL_GGUF, DATASET_GGUF, false));

    TEST_ASSERT_NOT_NULL(dataset.get(), "valid GGUF file");
    TEST_LOG_SUCCESS("Successfully loaded valid GGUF file");

    // Test basic properties
    TEST_ASSERT(validate_basic_dataset_properties(dataset.get()), "Basic dataset properties should be valid");
    display_dataset_summary(dataset.get(), "GGUF dataset");

    // Test metadata access
    TEST_ASSERT(test_metadata_access(dataset.get()), "Metadata access should work");

    // Test tensor access
    struct ggml_tensor* tensor = llama_dataset_sequence_tensor(dataset.get(), 0);
    TEST_ASSERT_NOT_NULL(tensor, "first sequence tensor");
    TEST_LOG_SUCCESS("Tensor access works");
}

// Test conversion between formats
void test_format_conversion();
void test_format_conversion() {
    TEST_LOG_SECTION("Testing format conversion");

    // Load a GGUF dataset
    TestDatasetGuard dataset(load_test_dataset(TEST_DATA_SMALL_GGUF, DATASET_GGUF, false));
    TEST_ASSERT_NOT_NULL(dataset.get(), "source GGUF dataset");

    // Test round-trip conversion
    TEST_ASSERT(test_dataset_conversion_roundtrip(dataset.get(), TEST_OUTPUT_GGUF), 
                "Round-trip conversion should preserve data");
    
    TEST_LOG_SUCCESS("Format conversion test passed");
}

// Test metadata access functions
void test_metadata_access_detailed();
void test_metadata_access_detailed() {
    TEST_LOG_SECTION("Testing metadata access functions");

    // Load a GGUF dataset
    TestDatasetGuard dataset(load_test_dataset(TEST_DATA_SMALL_GGUF, DATASET_GGUF, false));
    TEST_ASSERT_NOT_NULL(dataset.get(), "GGUF dataset for metadata testing");

    // Test metadata access
    TEST_ASSERT(test_metadata_access(dataset.get()), "Metadata access should work");

    // Test float metadata with default value
    float test_float = llama_dataset_get_metadata_float(dataset.get(), "test.float", -1.0f);
    TEST_ASSERT_EQUAL(-1.0f, test_float, "default float metadata");

    // Test null dataset handling
    TEST_ASSERT(test_null_dataset_handling(), "Null dataset handling should work");
    TEST_LOG_SUCCESS("Null dataset metadata handling works");

    TEST_LOG_SUCCESS("Metadata access test passed");
}

int main() {
    LLAMA_LOG_INFO("=== Running GGUF functionality tests ===\n");

    // Test GGUF factory functions
    test_gguf_factory();

    // Test format conversion
    test_format_conversion();

    // Test metadata access
    test_metadata_access_detailed();

    LLAMA_LOG_INFO("\n=== All GGUF functionality tests completed successfully! ===\n");
    return 0;
}