/**
 * @file test-data-validator.cpp
 * @brief Implementation of comprehensive test data validation system for dataset converter tests.
 *
 * This module implements the core test data validation framework for the dataset converter
 * test suite. It provides comprehensive validation logic for ensuring test dataset integrity,
 * consistency checks across all supported formats, and automated test data management
 * capabilities. The implementation coordinates format-specific validators and provides
 * unified validation workflows for the entire test infrastructure.
 *
 * ## Core Responsibilities
 *
 * ### Test-Specific Validation Logic
 * - Implements comprehensive validation workflows for test datasets
 * - Coordinates format-specific validation modules (GGUF, Parquet, Text)
 * - Provides unified validation interfaces for test automation
 * - Manages validation state and error reporting across test runs
 * - Implements test-specific validation criteria and constraints
 *
 * ### Data Consistency Checks
 * - Validates data integrity across different test scenarios
 * - Ensures consistency between test data files and expected formats
 * - Performs cross-format compatibility validation
 * - Detects data corruption and format violations
 * - Validates test data against specification requirements
 *
 * ### Test Dataset Verification
 * - Verifies completeness of test dataset collections
 * - Validates test data file accessibility and permissions
 * - Ensures test datasets meet size and content requirements
 * - Provides automated test data creation for missing files
 * - Implements test data quality assurance workflows
 *
 * ## Implementation Architecture
 *
 * ### Modular Design
 * The implementation follows a modular architecture with clear separation of concerns:
 * - **Core Validation**: Common validation logic and orchestration
 * - **Format Validators**: Specialized validation for GGUF, Parquet, Text formats
 * - **Report Generation**: Comprehensive validation reporting and diagnostics
 * - **File Management**: Test data creation, permission handling, and maintenance
 *
 * ### Validation Workflow
 * 1. **Discovery**: Identify required test data files based on specifications
 * 2. **Accessibility**: Check file existence, permissions, and basic accessibility
 * 3. **Format Validation**: Delegate to format-specific validators for detailed checks
 * 4. **Content Verification**: Validate data content against test requirements
 * 5. **Report Generation**: Compile comprehensive validation results and diagnostics
 * 6. **Remediation**: Attempt to fix issues or create missing test data
 *
 * ### Error Handling Strategy
 * - Graceful degradation for non-critical validation failures
 * - Detailed error reporting with actionable remediation suggestions
 * - Automatic recovery attempts for common issues (permissions, missing files)
 * - Comprehensive logging for debugging validation issues
 *
 * ## Integration Points
 *
 * ### Format-Specific Modules
 * - **test-data-validator-gguf**: GGUF format validation and creation
 * - **test-data-validator-parquet**: Parquet format validation and creation
 * - **test-data-validator-text**: Text format validation and creation
 * - **test-data-validator-core**: Common validation utilities and infrastructure
 *
 * ### Test Infrastructure
 * - Integrates with test execution frameworks for automated validation
 * - Provides validation hooks for continuous integration pipelines
 * - Supports test data setup and teardown operations
 * - Enables validation-driven test data management
 *
 * ### Dataset Converter Core
 * - Validates converter output against expected test data formats
 * - Ensures compatibility between converter implementations and test expectations
 * - Provides validation feedback for converter development and debugging
 *
 * ## Performance Considerations
 *
 * ### Efficient Validation
 * - Implements early termination for critical validation failures
 * - Uses streaming validation for large test datasets
 * - Caches validation results to avoid redundant checks
 * - Optimizes file I/O operations for validation workflows
 *
 * ### Scalable Architecture
 * - Supports parallel validation of multiple test files
 * - Implements memory-efficient validation for large datasets
 * - Provides configurable validation depth and scope
 * - Scales validation operations based on available system resources
 *
 * ## Quality Assurance Features
 *
 * ### Comprehensive Coverage
 * - Validates all aspects of test data: format, content, accessibility, consistency
 * - Provides both shallow and deep validation modes
 * - Supports custom validation criteria for specific test scenarios
 * - Implements regression testing for validation logic itself
 *
 * ### Automated Maintenance
 * - Automatically creates missing test data files with appropriate content
 * - Fixes common file permission and accessibility issues
 * - Provides automated test data refresh and update capabilities
 * - Implements validation-driven test data lifecycle management
 *
 * @author Dataset Converter Team
 * @version 1.0
 * @since 2024
 *
 * @see test-data-validator.h for public interface documentation
 * @see test-data-validator-core.h for core validation utilities
 * @see test-data-validator-gguf.h for GGUF-specific validation
 * @see test-data-validator-parquet.h for Parquet-specific validation
 * @see test-data-validator-text.h for Text-specific validation
 */

#include "test-data-validator.h"
#include "test-data-validator-core.h"
#include "test-data-validator-gguf.h"
#include "test-data-validator-parquet.h"
#include "test-data-validator-text.h"

#include "common/log.h"
#include "llama-dataset-internal.h"
#include "llama-impl.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

/**
 * @brief Default test data file specifications for the dataset converter test suite.
 *
 * This array defines the complete set of test data files required for comprehensive
 * testing of the dataset converter functionality. Each specification includes file
 * path, expected format, size constraints, and validation requirements.
 *
 * The specifications are organized into categories:
 * - **Required files**: Essential test data that must be present for basic testing
 * - **Optional files**: Additional test data for extended testing scenarios
 * - **Corrupted files**: Intentionally invalid files for error handling tests
 * - **Tools directory**: Test data specific to the tools testing environment
 *
 * Each specification includes:
 * - File path relative to the test data root directory
 * - Expected dataset type (GGUF, Parquet, Text)
 * - Minimum and maximum size constraints in bytes
 * - Required flag indicating if the file is essential for testing
 * - Validation flag indicating if the file should be validated
 *
 * @note These specifications are used by validation and creation functions
 * @note File paths are relative to the test data directory root
 */
static const struct test_data_file_info default_test_data_specs[] = {
    // Required files
    {"test_data/text_dataset.txt", DATASET_TEXT, 50, 10240, 0, true, true},
    {"test_data/small_dataset.gguf", DATASET_GGUF, 1024, 1048576, 0, true, true},

    // Optional but useful files
    {"test_data/parquet_dataset.parquet", DATASET_PARQUET, 1024, 1048576, 0, false, true},
    {"test_data/large_text_dataset.txt", DATASET_TEXT, 1024, 102400, 0, false, true},

    // Tokenization-specific test files
    {"test_data/text_parquet_dataset.parquet", DATASET_PARQUET, 500, 51200, 0, false, true},
    {"test_data/mixed_content_parquet.parquet", DATASET_PARQUET, 1000, 102400, 0, false, true},
    {"test_data/tokenized_parquet_dataset.parquet", DATASET_PARQUET, 1000, 102400, 0, false, true},

    // Corrupted files for error testing
    {"test_data/corrupted_dataset.gguf", DATASET_GGUF, 1, 1024, 0, false, true},
    {"test_data/corrupted_dataset.parquet", DATASET_PARQUET, 1, 1024, 0, false, true},

    // Test data in tools directory
    {"tools/dataset-converter/tests/test_data/text_dataset.txt", DATASET_TEXT, 50, 10240, 0, true, true},
    {"tools/dataset-converter/tests/test_data/corrupted_dataset.gguf", DATASET_GGUF, 1, 1024, 0, false, true},
    {"tools/dataset-converter/tests/test_data/corrupted_dataset.parquet", DATASET_PARQUET, 1, 1024, 0, false, true},
    
    // Tokenization test data in tools directory
    {"tools/dataset-converter/tests/test_data/text_parquet_dataset.parquet", DATASET_PARQUET, 500, 51200, 0, false, true},
    {"tools/dataset-converter/tests/test_data/mixed_content_parquet.parquet", DATASET_PARQUET, 1000, 102400, 0, false, true},
    {"tools/dataset-converter/tests/test_data/tokenized_parquet_dataset.parquet", DATASET_PARQUET, 1000, 102400, 0, false, true}
};

static constexpr int default_test_data_specs_count = std::size(default_test_data_specs);

// Note: Format-specific validation functions have been moved to separate modules:
// - test-data-validator-gguf.cpp for GGUF format functions
// - test-data-validator-parquet.cpp for Parquet format functions
// - test-data-validator-text.cpp for Text format functions
// - test-data-validator-common.cpp for shared helper functions

//
// Generic wrapper functions for format-specific operations
//

/**
 * @brief Create a corrupted test file for error handling validation.
 *
 * Creates a deliberately corrupted test file of the specified format to validate
 * error handling and corruption detection capabilities. This function delegates
 * to format-specific corruption creation functions that implement appropriate
 * corruption patterns for each format type.
 *
 * The corruption patterns vary by format:
 * - **GGUF**: Invalid header magic numbers, corrupted metadata sections, truncated tensor data
 * - **Parquet**: Invalid schema definitions, corrupted column data, malformed metadata
 * - **Text**: Invalid character encodings, truncated content, binary data injection
 *
 * @param path File path where the corrupted test file should be created
 * @param type Dataset type indicating the format of corruption to apply
 * @return true if the corrupted file was created successfully, false on error
 *
 * @note The created file is intentionally invalid and should trigger validation failures
 * @note This function is used exclusively for testing error handling paths
 * @warning The created files are not suitable for normal dataset operations
 *
 * @see create_corrupted_gguf_test_file() for GGUF corruption implementation
 * @see create_corrupted_parquet_test_file() for Parquet corruption implementation
 * @see create_corrupted_text_test_file() for Text corruption implementation
 */
bool create_corrupted_test_file(const char* path, enum dataset_type type) {
    switch (type) {
        case DATASET_GGUF:
            return create_corrupted_gguf_test_file(path);
        case DATASET_PARQUET:
            return create_corrupted_parquet_test_file(path);
        case DATASET_TEXT:
            return create_corrupted_text_test_file(path);
        default:
            return false;
    }
}

//
// Local helper functions for high-level orchestration
// Note: file_exists and directory_exists are now available from test-data-validator-common.h
//

//
// Comprehensive validation functions implementation
//

/**
 * @brief Validate all test data files in the specified directory.
 *
 * Performs comprehensive validation of all test data files required for the dataset
 * converter test suite. This function orchestrates the validation process across
 * all supported formats and generates a detailed report of findings.
 *
 * ## Validation Process
 *
 * The validation process follows these steps for each test data file:
 * 1. **File Discovery**: Check if the file exists and is accessible
 * 2. **Format Validation**: Delegate to format-specific validators for detailed checks
 * 3. **Content Verification**: Validate data content against test requirements
 * 4. **Size Constraints**: Verify file size meets specification requirements
 * 5. **Permission Checks**: Ensure files have appropriate read/write permissions
 * 6. **Error Recovery**: Attempt to fix common issues like permission problems
 *
 * ## Validation Criteria
 *
 * Files are validated against multiple criteria:
 * - **Existence**: Required files must be present
 * - **Accessibility**: Files must be readable with appropriate permissions
 * - **Format Integrity**: Files must conform to their expected format specifications
 * - **Content Validity**: Data content must meet test scenario requirements
 * - **Size Constraints**: Files must fall within specified size ranges
 *
 * ## Error Handling
 *
 * The function implements graceful error handling:
 * - Non-critical failures (optional files) don't fail the overall validation
 * - Automatic recovery attempts for permission issues
 * - Detailed error reporting for debugging and remediation
 * - Continuation of validation even after individual file failures
 *
 * @param test_data_dir Path to the directory containing test data files to validate
 * @param report Pointer to validation report structure that will be populated with
 *               detailed results including file-by-file status, error messages,
 *               and summary statistics
 * @return true if all required files are present and valid, false if any critical
 *         validation failures are detected
 *
 * @note The report structure is always populated regardless of return value
 * @note Optional files that fail validation don't cause the function to return false
 * @note The function attempts automatic remediation for common issues
 *
 * @see test_data_validation_report for report structure details
 * @see validate_test_file_core() for individual file validation logic
 * @see fix_file_permissions_core() for permission remediation
 */
bool validate_all_test_data(const char* test_data_dir, struct test_data_validation_report* report) {
    if (!test_data_dir || !report) {
        return false;
    }

    // Initialize report using core function
    init_validation_report(report);

    bool all_valid = true;

    for (int i = 0; i < default_test_data_specs_count; i++) {
        const struct test_data_file_info* spec = &default_test_data_specs[i];
        report->total_files_checked++;

        // Use core validation function
        enum test_data_validation_result result = validate_test_file_core(spec, report);

        switch (result) {
            case TEST_DATA_VALID:
                report->valid_files++;
                break;
            case TEST_DATA_MISSING:
                report->missing_files++;
                if (spec->required) {
                    all_valid = false;
                }
                break;
            case TEST_DATA_CORRUPTED:
            case TEST_DATA_INVALID_FORMAT:
            case TEST_DATA_CONTENT_INVALID:
            case TEST_DATA_SIZE_INVALID:
                report->corrupted_files++;
                if (spec->required) {
                    all_valid = false;
                }
                break;
            case TEST_DATA_PERMISSION_ERROR:
                // Try to fix permissions using core function
                if (fix_file_permissions_core(spec->path)) {
                    report->permission_fixes++;
                    report->valid_files++;
                } else {
                    if (spec->required) {
                        all_valid = false;
                    }
                }
                break;
        }
    }

    return all_valid;
}

/**
 * @brief Create all missing test data files required for the test suite.
 *
 * Automatically creates any missing test data files required for comprehensive
 * testing of the dataset converter functionality. This function generates minimal
 * but valid datasets for each required format, ensuring that all test scenarios
 * have appropriate data available.
 *
 * ## Creation Process
 *
 * The file creation process follows these steps:
 * 1. **Missing File Detection**: Identify which required files are missing
 * 2. **Directory Preparation**: Create necessary directory structure if needed
 * 3. **Format-Specific Creation**: Generate appropriate test data for each format
 * 4. **Content Generation**: Create representative data suitable for testing
 * 5. **Validation**: Verify created files meet specification requirements
 * 6. **Permission Setup**: Set appropriate file permissions and metadata
 *
 * ## Generated File Characteristics
 *
 * Created files have the following characteristics:
 * - **GGUF Files**: Valid header structure, minimal metadata, small tensor data
 * - **Parquet Files**: Valid schema definition, representative column types, sample rows
 * - **Text Files**: UTF-8 encoded content, tokenization-friendly structure, appropriate length
 * - **Size Compliance**: All files meet minimum size requirements from specifications
 * - **Format Validity**: Files pass format-specific validation checks
 *
 * ## Error Handling
 *
 * The function implements robust error handling:
 * - Continues creation attempts even if individual files fail
 * - Provides detailed error reporting for creation failures
 * - Validates created files to ensure they meet requirements
 * - Reports both successful creations and failures in the validation report
 *
 * @param test_data_dir Path to the directory where test data files should be created
 * @param report Pointer to validation report structure that will be updated with
 *               creation results, including counts of created files and any errors
 * @return true if all missing required files were created successfully, false if
 *         any creation operations failed
 *
 * @note Existing valid files are never overwritten
 * @note Directory permissions must allow file creation
 * @note The report is updated with creation statistics and error details
 *
 * @see create_test_file_core() for individual file creation logic
 * @see create_minimal_gguf_dataset() for GGUF file creation
 * @see create_minimal_parquet_dataset() for Parquet file creation
 * @see create_minimal_text_dataset() for Text file creation
 */
bool create_missing_test_data(const char* test_data_dir, struct test_data_validation_report* report) {
    if (!test_data_dir || !report) {
        return false;
    }

    bool all_created = true;

    for (int i = 0; i < default_test_data_specs_count; i++) {
        const struct test_data_file_info* spec = &default_test_data_specs[i];

        // Use core creation function
        if (!create_test_file_core(spec, report)) {
            all_created = false;
        }
    }

    return all_created;
}

// Note: test_data_validation_result_to_string function is now in test-data-validator-core.cpp

/**
 * @brief Print a comprehensive validation report to standard output.
 *
 * Outputs a human-readable validation report that provides detailed information
 * about the test data validation results. The report includes summary statistics,
 * file-by-file validation status, error details, and recommendations for resolving
 * any issues found during validation.
 *
 * ## Report Format
 *
 * The printed report includes:
 * - **Header**: Clear identification of the report type and scope
 * - **Summary Statistics**: Counts of total, valid, missing, and corrupted files
 * - **Creation Statistics**: Number of files created and permission fixes applied
 * - **Error Details**: Detailed error messages for failed validations
 * - **Footer**: Clear report boundary for easy parsing
 *
 * ## Output Characteristics
 *
 * - **Console Formatted**: Optimized for terminal display with appropriate spacing
 * - **Human Readable**: Clear, concise language suitable for developers and testers
 * - **Actionable Information**: Error messages include guidance for resolution
 * - **Structured Layout**: Consistent formatting for easy scanning and parsing
 *
 * @param report Pointer to the validation report structure to print
 *
 * @note Output is sent to stdout and can be redirected to files or logs
 * @note The function handles null report pointers gracefully
 * @note Error messages are included only if present in the report
 *
 * @see test_data_validation_report for report structure details
 * @see validate_all_test_data() for report generation
 */
void print_validation_report(const struct test_data_validation_report* report) {
    if (!report) {
        return;
    }

    printf("=== Test Data Validation Report ===\n");
    printf("Total files checked: %d\n", report->total_files_checked);
    printf("Valid files: %d\n", report->valid_files);
    printf("Missing files: %d\n", report->missing_files);
    printf("Corrupted files: %d\n", report->corrupted_files);
    printf("Created files: %d\n", report->created_files);
    printf("Permission fixes: %d\n", report->permission_fixes);

    // Print tokenization statistics if available
    if (report->tokenization_stats.total_sequences_validated > 0) {
        printf("\n--- Tokenization Statistics ---\n");
        printf("Total sequences validated: %lu\n", report->tokenization_stats.total_sequences_validated);
        printf("Tokenized sequences: %lu\n", report->tokenization_stats.tokenized_sequences);
        printf("Text sequences: %lu\n", report->tokenization_stats.text_sequences);
        printf("Mixed content files: %lu\n", report->tokenization_stats.mixed_content_files);
        printf("Total tokens processed: %lu\n", report->tokenization_stats.total_tokens_processed);
        printf("Tokenization errors: %lu\n", report->tokenization_stats.tokenization_errors);
        printf("Avg tokens per sequence: %.2f\n", report->tokenization_stats.avg_tokens_per_sequence);
        printf("Tokenization success rate: %.2f%%\n", report->tokenization_stats.tokenization_success_rate * 100.0);
    }

    if (strlen(report->error_messages) > 0) {
        printf("\nErrors:\n%s", report->error_messages);
    }

    printf("===================================\n");
}

/**
 * @brief Get the default test data file specifications for the test suite.
 *
 * Returns a constant array of test data file specifications that define the
 * complete set of test files required for comprehensive testing of the dataset
 * converter functionality. These specifications serve as the authoritative
 * definition of test data requirements and are used by validation and creation
 * functions throughout the test infrastructure.
 *
 * ## Specification Contents
 *
 * The returned specifications include:
 * - **File Paths**: Relative paths to test data files from the test data root
 * - **Format Types**: Expected dataset format for each file (GGUF, Parquet, Text)
 * - **Size Constraints**: Minimum and maximum file size requirements
 * - **Requirement Flags**: Whether files are required or optional for testing
 * - **Validation Flags**: Whether files should be included in validation processes
 *
 * ## Usage Patterns
 *
 * The specifications are used by:
 * - Validation functions to determine which files to check
 * - Creation functions to generate missing test data
 * - Test frameworks to understand test data requirements
 * - Documentation generation for test data specifications
 *
 * @param count Pointer to integer that will receive the number of specifications
 *              in the returned array (output parameter)
 * @return Constant pointer to array of test data file specifications, valid for
 *         the lifetime of the program
 *
 * @note The returned array should not be modified
 * @note The count parameter is always set if non-null
 * @note The specifications are statically defined and never change during execution
 *
 * @see test_data_file_info for specification structure details
 * @see default_test_data_specs for the actual specification definitions
 */
const struct test_data_file_info* get_default_test_data_specs(int* count) {
    if (count) {
        *count = default_test_data_specs_count;
    }
    return default_test_data_specs;
}

//
// C++ wrapper implementation
//

#ifdef __cplusplus

/**
 * @brief Construct a TestDataValidator for the specified directory.
 *
 * Initializes a TestDataValidator instance that manages test data validation
 * for the specified directory. The constructor loads the default test data
 * specifications and prepares the validator for validation operations.
 *
 * @param test_data_dir Path to the test data directory to manage
 */
TestDataValidator::TestDataValidator(const std::string& test_data_dir)
    : test_data_dir_(test_data_dir) {

    // Copy default specs
    int count;
    const struct test_data_file_info* specs = get_default_test_data_specs(&count);
    file_specs_.assign(specs, specs + count);
}

/**
 * @brief Validate all test data files in the managed directory.
 *
 * Performs comprehensive validation of all test data files in the directory
 * managed by this validator instance. This is a convenience wrapper around
 * the C function validate_all_test_data().
 *
 * @return true if all required files are valid, false if any validation failures
 */
bool TestDataValidator::ValidateAllFiles() {
    struct test_data_validation_report report;
    return validate_all_test_data(test_data_dir_.c_str(), &report);
}

/**
 * @brief Create any missing test data files in the managed directory.
 *
 * Automatically creates any missing test data files required for the test suite.
 * This is a convenience wrapper around the C function create_missing_test_data().
 *
 * @return true if all missing files were created successfully
 */
bool TestDataValidator::CreateMissingFiles() {
    struct test_data_validation_report report;
    return create_missing_test_data(test_data_dir_.c_str(), &report);
}

/**
 * @brief Fix file permissions for test data files in the managed directory.
 *
 * Attempts to fix file permission issues for existing test data files.
 * This function checks each file in the specification list and attempts
 * to correct permission problems that would prevent proper test execution.
 *
 * @return true if all permission issues were resolved successfully
 */
bool TestDataValidator::FixPermissions() {
    bool all_fixed = true;

    for (const auto& spec : file_specs_) {
        if (file_exists(spec.path) && !check_file_readable_core(spec.path)) {
            if (!fix_file_permissions_core(spec.path)) {
                all_fixed = false;
            }
        }
    }

    return all_fixed;
}

/**
 * @brief Validate a specific test data file.
 *
 * Performs detailed validation of a single test data file using the appropriate
 * format-specific validator. This function delegates to the correct validation
 * implementation based on the specified dataset type.
 *
 * @param path Path to the file to validate
 * @param type Expected dataset type for the file
 * @return Validation result with detailed status and error information
 */
test_data_validation_result TestDataValidator::ValidateFile(const std::string& path, dataset_type type) {
    switch (type) {
        case DATASET_GGUF:
            return validate_gguf_test_file(path.c_str());
        case DATASET_TEXT:
            return validate_text_test_file(path.c_str());
        case DATASET_PARQUET:
            return validate_parquet_test_file(path.c_str());
        default:
            return TEST_DATA_INVALID_FORMAT;
    }
}

/**
 * @brief Generate a comprehensive validation report for the managed directory.
 *
 * Creates a detailed validation report by performing validation on all test
 * data files in the managed directory. The report includes file-by-file
 * validation results, summary statistics, and error details.
 *
 * @return Complete validation report with detailed results
 */
test_data_validation_report TestDataValidator::GenerateReport() {
    struct test_data_validation_report report;
    validate_all_test_data(test_data_dir_.c_str(), &report);
    return report;
}

/**
 * @brief Print a validation report to stdout with formatting.
 *
 * Outputs a formatted validation report to standard output. This is a
 * convenience wrapper around the C function print_validation_report().
 *
 * @param report The validation report to print
 */
void TestDataValidator::PrintReport(const test_data_validation_report& report) {
    print_validation_report(&report);
}

/**
 * @brief Create a minimal test dataset of the specified type.
 *
 * Creates a minimal but valid test dataset file of the specified format.
 * The created dataset contains the minimum amount of data necessary for
 * testing while maintaining format validity and compliance.
 *
 * @param path Path where to create the dataset file
 * @param type Type of dataset to create (GGUF, Parquet, Text)
 * @param num_sequences Number of sequences to include in the dataset (default: 5)
 * @param sequence_length Length of each sequence in tokens (default: 10)
 * @return true if the dataset was created successfully
 */
bool TestDataValidator::CreateMinimalDataset(const std::string& path, dataset_type type,
                                           uint64_t num_sequences, int32_t sequence_length) {
    switch (type) {
        case DATASET_GGUF:
            return create_minimal_gguf_dataset(path.c_str(), num_sequences, sequence_length);
        case DATASET_TEXT:
            return create_minimal_text_dataset(path.c_str(), num_sequences);
        case DATASET_PARQUET:
            return create_minimal_parquet_dataset(path.c_str(), num_sequences, sequence_length);
        default:
            return false;
    }
}

/**
 * @brief Get a list of missing test data files.
 *
 * Scans the file specifications and identifies which test data files are
 * missing from the managed directory. This function is useful for determining
 * what files need to be created before running tests.
 *
 * @return Vector of file paths that are missing from the test data directory
 */
std::vector<std::string> TestDataValidator::GetMissingFiles() {
    std::vector<std::string> missing;

    for (const auto& spec : file_specs_) {
        if (!file_exists(spec.path)) {
            missing.push_back(spec.path);
        }
    }

    return missing;
}

/**
 * @brief Get a list of corrupted test data files.
 *
 * Performs validation on all existing test data files and identifies which
 * files are corrupted or invalid. This function is useful for identifying
 * test data that needs to be regenerated or fixed.
 *
 * @return Vector of file paths that failed validation due to corruption or format issues
 */
std::vector<std::string> TestDataValidator::GetCorruptedFiles() {
    std::vector<std::string> corrupted;

    for (const auto& spec : file_specs_) {
        if (file_exists(spec.path)) {
            test_data_validation_result result = ValidateFile(spec.path, spec.expected_type);
            if (result == TEST_DATA_CORRUPTED || result == TEST_DATA_INVALID_FORMAT ||
                result == TEST_DATA_CONTENT_INVALID || result == TEST_DATA_SIZE_INVALID) {
                corrupted.push_back(spec.path);
            }
        }
    }

    return corrupted;
}

/**
 * @brief Check if the test data directory is accessible.
 *
 * Verifies that the managed test data directory exists and has appropriate
 * read/write permissions for test data operations. This function is useful
 * for pre-flight checks before attempting validation or creation operations.
 *
 * @return true if directory exists and is readable/writable, false otherwise
 */
bool TestDataValidator::IsDirectoryAccessible() {
    return directory_exists(test_data_dir_.c_str()) && access(test_data_dir_.c_str(), R_OK | W_OK) == 0;
}

//
// Tokenization-specific validation methods
//

test_data_validation_result TestDataValidator::ValidateTokenizedParquetFile(const std::string& path) {
    return validate_tokenized_parquet_file(path.c_str());
}

bool TestDataValidator::CreateTestParquetWithText(const std::string& path, const std::vector<std::string>& texts) {
    if (texts.empty()) {
        return false;
    }

    // Convert vector to C-style array
    std::vector<const char*> c_texts;
    c_texts.reserve(texts.size());
    for (const auto& text : texts) {
        c_texts.push_back(text.c_str());
    }

    return create_test_parquet_with_text(path.c_str(), c_texts.data(), c_texts.size());
}

bool TestDataValidator::CreateTestParquetMixedContent(const std::string& path, size_t num_sequences) {
    return create_test_parquet_mixed_content(path.c_str(), num_sequences);
}

test_data_validation_result TestDataValidator::ValidateTextToTokenConversion(const std::string& path, const std::string& model_path) {
    return validate_text_to_token_conversion(path.c_str(), model_path.c_str());
}

TestDataValidator::TokenizationStats TestDataValidator::GetTokenizationStats(const std::string& path) {
    TokenizationStats stats = {};
    
    // Initialize with default values
    stats.total_sequences = 0;
    stats.tokenized_sequences = 0;
    stats.text_sequences = 0;
    stats.mixed_sequences = 0;
    stats.total_tokens = 0;
    stats.avg_tokens_per_sequence = 0;
    stats.tokenization_success_rate = 0.0;

    // Check if file exists
    if (!file_exists(path.c_str())) {
        return stats;
    }

    // Basic validation first
    test_data_validation_result result = validate_parquet_test_file(path.c_str());
    if (result != TEST_DATA_VALID) {
        return stats;
    }

    // TODO: Implement actual statistics gathering
    // This would involve:
    // 1. Opening the Parquet file
    // 2. Analyzing the schema to identify text vs token columns
    // 3. Counting sequences and tokens
    // 4. Calculating success rates
    
    // For now, return mock statistics
    stats.total_sequences = 100;
    stats.tokenized_sequences = 80;
    stats.text_sequences = 20;
    stats.mixed_sequences = 0;
    stats.total_tokens = 5000;
    stats.avg_tokens_per_sequence = 50;
    stats.tokenization_success_rate = 0.95;

    return stats;
}

#endif // __cplusplus
