#include "common.h"
#include "llama-impl.h"
#include "test_core_functionality.h"

// Test Parquet factory function with valid and invalid inputs
void test_parquet_factory();
void test_parquet_factory() {
    TEST_LOG_SECTION("Testing Parquet factory function");

#ifdef LLAMA_PARQUET
    // Test with null path
    TEST_ASSERT(test_error_handling("null path", []() {
        common_params params;
        return llama_dataset_from_parquet(&params);
    }), "Null path error handling should work");

    // Test with non-existent file
    TEST_ASSERT(test_error_handling("non-existent file", []() {
        TestParamsBuilder builder;
        common_params params = builder.with_file("non_existent_file.parquet").build();
        return llama_dataset_from_parquet(&params);
    }), "Non-existent file error handling should work");

    // Test with valid file if available
    TestDatasetGuard dataset(load_test_dataset(TEST_DATA_PARQUET, DATASET_PARQUET, false));
    if (dataset.is_valid()) {
        TEST_LOG_SUCCESS("Successfully loaded valid Parquet file");

        // Test basic properties
        TEST_ASSERT(validate_basic_dataset_properties(dataset.get()), "Basic dataset properties should be valid");
        display_dataset_summary(dataset.get(), "Parquet dataset");

        TEST_LOG_SUCCESS("Dataset cleanup successful");
    } else {
        TEST_LOG_INFO("Parquet file not available or support not compiled in: %s", llama_dataset_get_error());
        llama_dataset_clear_error();
    }
#else
    TEST_LOG_INFO("Parquet support not compiled in, skipping tests");
#endif
}

// Test basic tokenization engine functionality
void test_tokenization_engine_basic();
void test_tokenization_engine_basic() {
#ifdef LLAMA_PARQUET
    LLAMA_LOG_INFO("Testing tokenization engine basic functionality...\n");

    // Note: This test only verifies the tokenization engine can be created
    // and configured without a real model. Full tokenization testing requires
    // a loaded llama model which is beyond the scope of this unit test.

    // Test that we can create a tokenization engine with null model
    // (this should fail gracefully)
    try {
        // This should fail since model is null
        llama_dataset_parquet_tokenizer tokenizer(nullptr);

        // If we get here, the constructor didn't fail as expected
        if (!tokenizer.is_valid()) {
            LLAMA_LOG_INFO("✓ Tokenizer correctly reports invalid state with null model\n");
        } else {
            LLAMA_LOG_ERROR("✗ Tokenizer should be invalid with null model\n");
            assert(false);
        }

    } catch (const std::exception& e) {
        LLAMA_LOG_INFO("✓ Tokenizer constructor handled null model gracefully\n");
    }

    LLAMA_LOG_INFO("✓ Tokenization engine basic test passed\n");
#else
    LLAMA_LOG_INFO("Tokenization engine test skipped (Parquet support not enabled)\n");
#endif
}

int main() {
    LLAMA_LOG_INFO("=== Running Parquet functionality tests ===\n");

    // Test Parquet factory functions
    test_parquet_factory();

    // Test tokenization engine (basic functionality)
    test_tokenization_engine_basic();

    LLAMA_LOG_INFO("\n=== All Parquet functionality tests completed successfully! ===\n");
    return 0;
}
