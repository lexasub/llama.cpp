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
#include "test-data-validator-common.h"
#include "test-data-validator-core.h"
#include "test-data-validator-gguf.h"
#include "test-data-validator-parquet.h"
#include "test-data-validator-text.h"

#ifdef __cplusplus
extern "C" {
#endif

// Note: Common types and structures are now defined in test-data-validator-common.h

// Note: Format-specific validation and creation functions are now defined in:
// - test-data-validator-gguf.h for GGUF format functions
// - test-data-validator-parquet.h for Parquet format functions  
// - test-data-validator-text.h for Text format functions
// - test-data-validator-common.h for shared helper functions

//
// Generic wrapper functions
//

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

// Note: test_data_validation_result_to_string is now defined in test-data-validator-core.h

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
