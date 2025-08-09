#pragma once

/**
 * @file test_core_functionality.h
 * @brief Common test utilities and definitions for dataset converter tests
 *
 * This header provides shared utilities, mock structures, and common definitions
 * used across multiple test files in the dataset converter test suite.
 */

#include <gtest/gtest.h>
#include <string>
#include <cstdint>

// Forward declarations for dataset structures
struct llama_dataset;
struct gguf_context;
struct llama_model;

// Test result structure
struct TestResult {
    bool passed;
    std::string message;
    double duration_ms;
};

// Test constants for metadata keys (matching the main header definitions)
#define TRAINING_FORMAT_VERSION    "training.format.version"
#define TRAINING_FORMAT_SOURCE     "training.format.source"
#define TRAINING_DATASET_NAME      "training.dataset.name"
#define TRAINING_DATASET_DESCRIPTION "training.dataset.description"
#define TRAINING_SEQUENCE_COUNT    "training.sequence.count"
#define TRAINING_MAX_LENGTH        "dataset.max_length"
#define TRAINING_TOKENIZER         "training.tokenizer.gguf.model"
#define TRAINING_CREATION_TIME     "training.file.creation_date"

#define TEST_LOG_SUCCESS(msg) LLAMA_LOG_INFO("✓ %s\n", msg)
#define TEST_LOG_FAILURE(msg) LLAMA_LOG_ERROR("✗ %s\n", msg)
#define TEST_LOG_WARNING(msg, ...) LLAMA_LOG_WARN(msg, __VA_ARGS__)
// Mock dataset structure for testing
struct mock_dataset_context {
    bool has_metadata;
    std::string dataset_name;
    std::string dataset_description;
    int64_t sequence_count;
    float float_value;
    std::string string_number;
    std::string string_float;
};

// Test utility functions
namespace test_utils {
    /**
     * @brief Create a mock dataset with predefined metadata for testing
     * @return Pointer to mock dataset structure (caller must free)
     */
    struct llama_dataset* create_mock_dataset_with_metadata();

    /**
     * @brief Free a mock dataset created by create_mock_dataset_with_metadata
     * @param dataset Dataset to free
     */
    void free_mock_dataset(struct llama_dataset* dataset);

    /**
     * @brief Check if a string contains expected error message patterns
     * @param error_message The error message to check
     * @param expected_pattern Pattern that should be present
     * @return true if pattern is found, false otherwise
     */
    bool error_message_contains(const char* error_message, const char* expected_pattern);
}

// Test fixture base class for common setup/teardown
class DatasetTestBase : public ::testing::Test {
protected:
    void SetUp() override;
    void TearDown() override;

    // Helper methods for common test operations
    void clear_errors();
    bool has_error();
    const char* get_error_message();

    // Mock dataset for testing
    struct llama_dataset* test_dataset = nullptr;
};
