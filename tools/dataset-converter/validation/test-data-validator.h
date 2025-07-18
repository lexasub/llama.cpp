#pragma once

/**
 * @file test-data-validator.h
 * @brief Comprehensive test data validation system for dataset converter tests.
 *
 * This header provides functions to validate, create, and manage test data files
 * for the dataset converter test suite. It ensures all required test data files
 * are present, valid, and accessible.
 */

#include <stdint.h>

#include <string>
#include <vector>

#include "llama-dataset.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Test data validation result codes.
 */
enum test_data_validation_result {
    TEST_DATA_VALID = 0,                    // File is valid
    TEST_DATA_MISSING,                      // File does not exist
    TEST_DATA_CORRUPTED,                    // File exists but is corrupted
    TEST_DATA_INVALID_FORMAT,               // File format is invalid
    TEST_DATA_PERMISSION_ERROR,             // File permission issues
    TEST_DATA_SIZE_INVALID,                 // File size is invalid (too small/large)
    TEST_DATA_CONTENT_INVALID               // File content is invalid
};

/**
 * @brief Test data file information structure.
 */
struct test_data_file_info {
    const char* path;                       // File path
    enum dataset_type expected_type;        // Expected dataset type
    uint64_t min_size_bytes;               // Minimum expected file size
    uint64_t max_size_bytes;               // Maximum expected file size
    uint64_t expected_sequences;           // Expected number of sequences (0 = any)
    bool required;                         // Whether this file is required
    bool create_if_missing;                // Whether to create if missing
};

/**
 * @brief Test data validation report structure.
 */
struct test_data_validation_report {
    int total_files_checked;
    int valid_files;
    int missing_files;
    int corrupted_files;
    int created_files;
    int permission_fixes;
    char error_messages[4096];             // Concatenated error messages
};

//
// Core validation functions
//

/**
 * @brief Validate a GGUF test data file.
 *
 * @param path Path to the GGUF file
 * @return Validation result code
 */
enum test_data_validation_result validate_gguf_test_file(const char* path);

/**
 * @brief Validate a text test data file.
 *
 * @param path Path to the text file
 * @return Validation result code
 */
enum test_data_validation_result validate_text_test_file(const char* path);

/**
 * @brief Validate a Parquet test data file.
 *
 * @param path Path to the Parquet file
 * @return Validation result code
 */
enum test_data_validation_result validate_parquet_test_file(const char* path);

/**
 * @brief Check if a file has proper permissions for testing.
 *
 * @param path Path to the file
 * @return true if permissions are correct, false otherwise
 */
bool check_file_permissions(const char* path);

/**
 * @brief Fix file permissions for testing.
 *
 * @param path Path to the file
 * @return true if permissions were fixed successfully, false otherwise
 */
bool fix_file_permissions(const char* path);

//
// Test data creation functions
//

/**
 * @brief Create a minimal valid GGUF test dataset.
 *
 * @param path Path where to create the GGUF file
 * @param num_sequences Number of sequences to create
 * @param sequence_length Length of each sequence
 * @return true on success, false on error
 */
bool create_minimal_gguf_dataset(const char* path, uint64_t num_sequences, int32_t sequence_length);

/**
 * @brief Create a minimal valid text test dataset.
 *
 * @param path Path where to create the text file
 * @param num_lines Number of lines to create
 * @return true on success, false on error
 */
bool create_minimal_text_dataset(const char* path, uint64_t num_lines);

/**
 * @brief Create a minimal valid Parquet test dataset.
 *
 * @param path Path where to create the Parquet file
 * @param num_sequences Number of sequences to create
 * @param sequence_length Length of each sequence
 * @return true on success, false on error
 */
bool create_minimal_parquet_dataset(const char* path, uint64_t num_sequences, int32_t sequence_length);

/**
 * @brief Create a corrupted test file for error handling tests.
 *
 * @param path Path where to create the corrupted file
 * @param type Type of dataset to corrupt
 * @return true on success, false on error
 */
bool create_corrupted_test_file(const char* path, enum dataset_type type);

//
// Comprehensive validation functions
//

/**
 * @brief Validate all test data files in a directory.
 *
 * @param test_data_dir Path to the test data directory
 * @param report Pointer to validation report structure (output)
 * @return true if all required files are valid, false otherwise
 */
bool validate_all_test_data(const char* test_data_dir, struct test_data_validation_report* report);

/**
 * @brief Create all missing test data files.
 *
 * @param test_data_dir Path to the test data directory
 * @param report Pointer to validation report structure (output)
 * @return true if all files were created successfully, false otherwise
 */
bool create_missing_test_data(const char* test_data_dir, struct test_data_validation_report* report);

/**
 * @brief Get a string representation of a validation result.
 *
 * @param result Validation result code
 * @return String representation of the result
 */
const char* test_data_validation_result_to_string(enum test_data_validation_result result);

/**
 * @brief Print a validation report to stdout.
 *
 * @param report Pointer to the validation report
 */
void print_validation_report(const struct test_data_validation_report* report);

/**
 * @brief Get the default test data file specifications.
 *
 * @param count Pointer to store the number of file specifications (output)
 * @return Array of test data file specifications
 */
const struct test_data_file_info* get_default_test_data_specs(int* count);

#ifdef __cplusplus
}

/**
 * @brief C++ wrapper class for test data validation.
 */
class TestDataValidator {
private:
    std::string test_data_dir_;
    std::vector<test_data_file_info> file_specs_;

public:
    explicit TestDataValidator(const std::string& test_data_dir);

    // Validation methods
    bool ValidateAllFiles();
    bool CreateMissingFiles();
    bool FixPermissions();

    // Individual file validation
    test_data_validation_result ValidateFile(const std::string& path, dataset_type type);

    // Report generation
    test_data_validation_report GenerateReport();
    void PrintReport(const test_data_validation_report& report);

    // File creation
    bool CreateMinimalDataset(const std::string& path, dataset_type type,
                             uint64_t num_sequences = 5, int32_t sequence_length = 10);

    // Utility methods
    std::vector<std::string> GetMissingFiles();
    std::vector<std::string> GetCorruptedFiles();
    bool IsDirectoryAccessible();
};

#endif // __cplusplus
