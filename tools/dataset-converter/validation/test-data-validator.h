#pragma once

/**
 * @file test-data-validator.h
 * @brief Comprehensive test data validation system for dataset converter tests.
 *
 * This module provides a complete test data validation framework for the dataset converter
 * test suite. It ensures data consistency, validates test datasets across all supported
 * formats (GGUF, Parquet, Text), and provides utilities for creating, managing, and
 * verifying test data integrity.
 *
 * ## Key Responsibilities
 *
 * ### Test Data Validation
 * - Validates test datasets for correctness and integrity
 * - Performs format-specific validation checks for GGUF, Parquet, and Text files
 * - Detects corrupted or malformed test data files
 * - Ensures test data meets expected specifications and constraints
 *
 * ### Test Data Management
 * - Creates missing test data files with appropriate content
 * - Generates minimal datasets for basic testing scenarios
 * - Manages test data file permissions and accessibility
 * - Provides comprehensive validation reporting and diagnostics
 *
 * ### Quality Assurance
 * - Ensures test data consistency across different test environments
 * - Validates test data against expected schemas and formats
 * - Provides detailed error reporting for validation failures
 * - Supports automated test data verification workflows
 *
 * ## Validation Criteria
 *
 * ### File-Level Validation
 * - File existence and accessibility checks
 * - File size and format validation
 * - Permission and ownership verification
 * - Basic corruption detection
 *
 * ### Format-Specific Validation
 * - **GGUF**: Header validation, metadata consistency, tensor data integrity
 * - **Parquet**: Schema validation, column type verification, data consistency
 * - **Text**: Encoding validation, content structure, tokenization compatibility
 *
 * ### Content Validation
 * - Data type consistency and range validation
 * - Sequence length and structure verification
 * - Cross-format compatibility checks
 * - Performance benchmark data validation
 *
 * ## Integration
 *
 * This module integrates with:
 * - `llama-dataset-validation.h`: Core validation infrastructure
 * - Format-specific validators: GGUF, Parquet, Text validation modules
 * - Test execution framework: Automated validation during test runs
 * - Dataset converter core: Validation of converter output
 *
 * ## Usage Patterns
 *
 * ### Basic Validation
 * ```c
 * struct test_data_validation_report report;
 * if (validate_all_test_data("/path/to/test/data", &report)) {
 *     // All test data is valid
 * } else {
 *     print_validation_report(&report);
 * }
 * ```
 *
 * ### Test Data Creation
 * ```c
 * if (!create_missing_test_data("/path/to/test/data", &report)) {
 *     // Handle creation failures
 * }
 * ```
 *
 * ### C++ Interface
 * ```cpp
 * TestDataValidator validator("/path/to/test/data");
 * if (!validator.ValidateAllFiles()) {
 *     auto report = validator.GenerateReport();
 *     validator.PrintReport(report);
 * }
 * ```
 *
 * @author Dataset Converter Team
 * @version 1.0
 * @since 2024
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
#include "test-data-validator-orchestration.h"
#include "test-data-validator-creation.h"
#include "test-data-validator-reporting.h"

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
 * Creates a deliberately corrupted test file to validate error handling and
 * corruption detection capabilities. The corruption type varies by format:
 * - GGUF: Invalid header magic, corrupted metadata, or truncated tensors
 * - Parquet: Invalid schema, corrupted column data, or malformed metadata
 * - Text: Invalid encoding, truncated content, or binary data injection
 *
 * @param path Path where to create the corrupted file
 * @param type Type of dataset to corrupt (DATASET_TYPE_GGUF, DATASET_TYPE_PARQUET, etc.)
 * @return true on success, false on error
 *
 * @note The created file should trigger validation failures and error handling paths
 * @warning This function creates intentionally invalid files for testing purposes only
 */
bool create_corrupted_test_file(const char* path, enum dataset_type type);

//
// Comprehensive validation functions
//

/**
 * @brief Validate all test data files in a directory.
 *
 * Performs comprehensive validation of all test data files in the specified directory.
 * This includes checking file existence, accessibility, format validity, and content
 * integrity. The validation process covers all supported formats and generates a
 * detailed report of findings.
 *
 * Validation steps performed:
 * 1. Directory accessibility and permission checks
 * 2. Required file existence verification
 * 3. Format-specific validation for each file type
 * 4. Content integrity and consistency checks
 * 5. Cross-file compatibility validation
 *
 * @param test_data_dir Path to the test data directory to validate
 * @param report Pointer to validation report structure (output) - contains detailed
 *               results including file-by-file status, error messages, and statistics
 * @return true if all required files are present and valid, false if any validation
 *         failures are detected
 *
 * @note The report structure is populated regardless of return value
 * @see test_data_validation_report for report structure details
 */
bool validate_all_test_data(const char* test_data_dir, struct test_data_validation_report* report);

/**
 * @brief Create all missing test data files.
 *
 * Creates any missing test data files required for the test suite. This function
 * generates minimal but valid datasets for each required format, ensuring that
 * all test scenarios have appropriate data available. The created files follow
 * standard specifications and contain representative data for testing.
 *
 * Creation process:
 * 1. Identifies missing files based on default test data specifications
 * 2. Creates directory structure if needed
 * 3. Generates format-appropriate test data for each missing file
 * 4. Validates created files to ensure they meet requirements
 * 5. Sets appropriate file permissions and metadata
 *
 * Generated file characteristics:
 * - GGUF: Valid header, minimal metadata, small tensor data
 * - Parquet: Valid schema, representative column types, sample rows
 * - Text: UTF-8 encoded, tokenization-friendly content, appropriate length
 *
 * @param test_data_dir Path to the test data directory where files will be created
 * @param report Pointer to validation report structure (output) - updated with
 *               creation results and any errors encountered
 * @return true if all missing files were created successfully, false if any
 *         creation operations failed
 *
 * @note Existing valid files are not overwritten
 * @note Directory permissions must allow file creation
 */
bool create_missing_test_data(const char* test_data_dir, struct test_data_validation_report* report);

// Note: test_data_validation_result_to_string is now defined in test-data-validator-core.h

/**
 * @brief Print a validation report to stdout.
 *
 * Outputs a human-readable validation report to standard output. The report
 * includes summary statistics, file-by-file validation results, error details,
 * and recommendations for resolving any issues found.
 *
 * Report format includes:
 * - Overall validation status and summary statistics
 * - File-by-file validation results with status indicators
 * - Detailed error messages for failed validations
 * - Warnings for potential issues or inconsistencies
 * - Recommendations for fixing validation failures
 *
 * @param report Pointer to the validation report to print
 *
 * @note Output is formatted for console display with appropriate indentation
 * @note Uses ANSI color codes if terminal supports them
 */
void print_validation_report(const struct test_data_validation_report* report);

/**
 * @brief Get the default test data file specifications.
 *
 * Returns an array of default test data file specifications that define the
 * required test files for the dataset converter test suite. These specifications
 * include file paths, expected formats, size constraints, and validation criteria.
 *
 * The specifications cover:
 * - Basic format examples for each supported type (GGUF, Parquet, Text)
 * - Edge case test files (empty, minimal, large)
 * - Error condition test files (corrupted, invalid format)
 * - Performance benchmark datasets
 * - Cross-format compatibility test files
 *
 * @param count Pointer to store the number of file specifications (output)
 * @return Constant array of test data file specifications, valid until program exit
 *
 * @note The returned array should not be modified
 * @note The count parameter is set to the number of specifications in the array
 * @see test_data_file_info for specification structure details
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
    /**
     * @brief Construct a TestDataValidator for the specified directory.
     * @param test_data_dir Path to the test data directory to manage
     */
    explicit TestDataValidator(const std::string& test_data_dir);

    // Validation methods
    
    /**
     * @brief Validate all test data files in the managed directory.
     * @return true if all files are valid, false if any validation failures
     */
    bool ValidateAllFiles();
    
    /**
     * @brief Create any missing test data files.
     * @return true if all missing files were created successfully
     */
    bool CreateMissingFiles();
    
    /**
     * @brief Fix file permissions for test data files.
     * @return true if permissions were fixed successfully
     */
    bool FixPermissions();

    // Individual file validation
    
    /**
     * @brief Validate a specific test data file.
     * @param path Path to the file to validate
     * @param type Expected dataset type for the file
     * @return Validation result with detailed status and error information
     */
    test_data_validation_result ValidateFile(const std::string& path, dataset_type type);

    // Report generation
    
    /**
     * @brief Generate a comprehensive validation report.
     * @return Complete validation report with file-by-file results
     */
    test_data_validation_report GenerateReport();
    
    /**
     * @brief Print a validation report to stdout with formatting.
     * @param report The validation report to print
     */
    void PrintReport(const test_data_validation_report& report);

    // File creation
    
    /**
     * @brief Create a minimal test dataset of the specified type.
     * @param path Path where to create the dataset file
     * @param type Type of dataset to create
     * @param num_sequences Number of sequences to include (default: 5)
     * @param sequence_length Length of each sequence (default: 10)
     * @return true if dataset was created successfully
     */
    bool CreateMinimalDataset(const std::string& path, dataset_type type,
                             uint64_t num_sequences = 5, int32_t sequence_length = 10);

    // Tokenization-specific validation
    test_data_validation_result ValidateTokenizedParquetFile(const std::string& path);
    bool CreateTestParquetWithText(const std::string& path, const std::vector<std::string>& texts);
    bool CreateTestParquetMixedContent(const std::string& path, size_t num_sequences);
    test_data_validation_result ValidateTextToTokenConversion(const std::string& path, const std::string& model_path);

    // Enhanced reporting with tokenization statistics
    struct TokenizationStats {
        uint64_t total_sequences;
        uint64_t tokenized_sequences;
        uint64_t text_sequences;
        uint64_t mixed_sequences;
        uint64_t total_tokens;
        uint64_t avg_tokens_per_sequence;
        double tokenization_success_rate;
    };
    
    TokenizationStats GetTokenizationStats(const std::string& path);

    // Utility methods
    
    /**
     * @brief Get a list of missing test data files.
     * @return Vector of file paths that are missing from the test data directory
     */
    std::vector<std::string> GetMissingFiles();
    
    /**
     * @brief Get a list of corrupted test data files.
     * @return Vector of file paths that failed validation due to corruption
     */
    std::vector<std::string> GetCorruptedFiles();
    
    /**
     * @brief Check if the test data directory is accessible.
     * @return true if directory exists and is readable/writable
     */
    bool IsDirectoryAccessible();
};

#endif // __cplusplus
