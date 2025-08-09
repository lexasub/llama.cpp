#pragma once

/**
 * @file test-data-validator-core.h
 * @brief Core validation algorithms and orchestration for test data validation system.
 *
 * This module provides the foundational validation infrastructure and core algorithms
 * that power the test data validation system. It serves as the central coordination
 * hub for all validation operations, managing validation workflows, orchestrating
 * format-specific validators, and providing shared validation utilities used across
 * the entire validation framework.
 *
 * ## Key Responsibilities
 *
 * ### Core Validation Algorithms
 * - Implements fundamental validation algorithms used across all format types
 * - Provides common validation patterns and reusable validation logic
 * - Manages validation state and result aggregation across multiple files
 * - Coordinates complex validation workflows involving multiple validation steps
 *
 * ### Validation Orchestration
 * - Coordinates validation operations across format-specific validator modules
 * - Manages validation report generation and error aggregation
 * - Provides unified validation interfaces that abstract format-specific details
 * - Handles validation workflow sequencing and dependency management
 *
 * ### Shared Validation Infrastructure
 * - Implements common file system operations (permission checks, accessibility)
 * - Provides standardized error handling and logging mechanisms
 * - Manages validation report structures and result formatting
 * - Offers utility functions for file specification validation and processing
 *
 * ### Cross-Format Validation Support
 * - Enables validation operations that span multiple file formats
 * - Provides consistency checks across different dataset types
 * - Supports validation of format conversion accuracy and integrity
 * - Manages validation of cross-format compatibility requirements
 *
 * ## Validation Architecture
 *
 * ### Validation Flow
 * 1. **Initialization**: Set up validation report structures and prepare validation context
 * 2. **File Discovery**: Identify files to validate based on specifications
 * 3. **Format Detection**: Determine appropriate validator for each file type
 * 4. **Validation Execution**: Delegate to format-specific validators while maintaining coordination
 * 5. **Result Aggregation**: Collect and consolidate validation results from all validators
 * 6. **Report Generation**: Compile comprehensive validation reports with detailed findings
 *
 * ### Error Handling Strategy
 * - Graceful degradation: Continue validation even when individual files fail
 * - Comprehensive error reporting: Capture detailed error information for debugging
 * - Error categorization: Classify errors by severity and type for appropriate handling
 * - Recovery mechanisms: Attempt to fix common issues (permissions, missing files)
 *
 * ## Core Algorithms
 *
 * ### File Validation Algorithm
 * ```
 * 1. Validate file specification structure
 * 2. Check file existence and accessibility
 * 3. Verify file permissions and ownership
 * 4. Delegate to format-specific validator
 * 5. Aggregate results and update validation report
 * ```
 *
 * ### Permission Management Algorithm
 * ```
 * 1. Check current file permissions
 * 2. Identify required permissions for test operations
 * 3. Apply necessary permission changes if authorized
 * 4. Verify permission changes were successful
 * 5. Log permission operations for audit trail
 * ```
 *
 * ### Validation Report Management
 * ```
 * 1. Initialize report structure with default values
 * 2. Track validation progress and statistics
 * 3. Accumulate errors and warnings from all validators
 * 4. Generate summary statistics and recommendations
 * 5. Format results for human-readable output
 * ```
 *
 * ## Integration Points
 *
 * This module integrates with:
 * - **Format-specific validators**: GGUF, Parquet, Text validation modules
 * - **File system layer**: Platform-specific file operations and permission management
 * - **Logging system**: Centralized error and diagnostic logging
 * - **Test framework**: Integration with automated testing workflows
 * - **Validation reporting**: Report generation and formatting systems
 *
 * ## Performance Considerations
 *
 * ### Optimization Strategies
 * - Parallel validation: Support for concurrent validation of multiple files
 * - Early termination: Stop validation on critical errors when appropriate
 * - Caching: Cache validation results for repeated operations
 * - Resource management: Efficient memory usage during large-scale validation
 *
 * ### Scalability Features
 * - Batch processing: Handle validation of large numbers of files efficiently
 * - Progress tracking: Provide feedback during long-running validation operations
 * - Resource limits: Respect system resource constraints during validation
 * - Incremental validation: Support for validating only changed files
 *
 * ## Usage Patterns
 *
 * ### Basic File Validation
 * ```c
 * struct test_data_validation_report report;
 * init_validation_report(&report);
 * 
 * struct test_data_file_info spec = {
 *     .path = "/path/to/test.gguf",
 *     .type = DATASET_TYPE_GGUF,
 *     .expected_size = 1024
 * };
 * 
 * enum test_data_validation_result result = validate_test_file_core(&spec, &report);
 * if (result != TEST_DATA_VALIDATION_SUCCESS) {
 *     log_validation_error_core(spec.path, result);
 * }
 * ```
 *
 * ### File Creation and Validation
 * ```c
 * if (!create_test_file_core(&spec, &report)) {
 *     // Handle creation failure
 * } else {
 *     // Validate the newly created file
 *     validate_test_file_core(&spec, &report);
 * }
 * ```
 *
 * ### Permission Management
 * ```c
 * if (!check_file_readable_core("/path/to/test.dat")) {
 *     if (!fix_file_permissions_core("/path/to/test.dat")) {
 *         // Handle permission fix failure
 *     }
 * }
 * ```
 *
 * @author Dataset Converter Team
 * @version 1.0
 * @since 2024
 * @see test-data-validator.h Main validation interface
 * @see test-data-validator-common.h Common types and structures
 */

#include "test-data-validator-common.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Core validation orchestration functions
//

/**
 * @brief Initialize a validation report structure with default values.
 *
 * Prepares a validation report structure for use by setting all fields to their
 * default values and allocating any necessary internal resources. This function
 * must be called before using a validation report structure with any other
 * validation functions.
 *
 * Initialization operations performed:
 * - Reset all counters (total_files, valid_files, invalid_files) to zero
 * - Clear all error message arrays and set error counts to zero
 * - Initialize timing information and performance metrics
 * - Set validation status flags to default states
 * - Prepare internal data structures for error tracking
 *
 * @param report Pointer to the validation report structure to initialize.
 *               Must not be NULL. The structure will be completely reset
 *               regardless of its previous state.
 *
 * @note This function is safe to call multiple times on the same report structure
 * @note No memory allocation is performed; the report structure must be allocated by caller
 * @warning Calling this function on a report with accumulated data will lose all previous results
 *
 * @see test_data_validation_report Structure definition and field descriptions
 */
void init_validation_report(struct test_data_validation_report* report);

/**
 * @brief Validate a single test file using the appropriate format-specific validator.
 *
 * This is the central validation coordination function that orchestrates the complete
 * validation process for a single test file. It performs preliminary checks, determines
 * the appropriate format-specific validator, delegates the detailed validation, and
 * aggregates the results into the validation report.
 *
 * Validation process flow:
 * 1. **Specification Validation**: Verify the file specification structure is valid
 * 2. **File System Checks**: Confirm file existence, accessibility, and basic properties
 * 3. **Permission Verification**: Ensure the file has appropriate read permissions
 * 4. **Format Detection**: Confirm the file type matches the specification
 * 5. **Delegation**: Call the appropriate format-specific validator (GGUF, Parquet, Text)
 * 6. **Result Processing**: Interpret validation results and update the report
 * 7. **Error Handling**: Log any errors and update error statistics
 *
 * Format-specific delegation:
 * - **GGUF files**: Delegates to `validate_gguf_test_file()` for header and tensor validation
 * - **Parquet files**: Delegates to `validate_parquet_test_file()` for schema and data validation
 * - **Text files**: Delegates to `validate_text_test_file()` for encoding and content validation
 * - **Unknown formats**: Returns appropriate error code without delegation
 *
 * @param spec Pointer to the file specification containing path, expected type,
 *             size constraints, and validation criteria. Must not be NULL and
 *             must contain valid file path and type information.
 * @param report Pointer to the validation report where results will be recorded.
 *               Must not be NULL. The report is updated with validation results,
 *               error messages, and statistics regardless of validation outcome.
 * @return Validation result code indicating the outcome:
 *         - TEST_DATA_VALIDATION_SUCCESS: File passed all validation checks
 *         - TEST_DATA_VALIDATION_FILE_NOT_FOUND: File does not exist at specified path
 *         - TEST_DATA_VALIDATION_PERMISSION_ERROR: Insufficient permissions to read file
 *         - TEST_DATA_VALIDATION_FORMAT_ERROR: File format doesn't match specification
 *         - TEST_DATA_VALIDATION_CORRUPTION_ERROR: File appears to be corrupted
 *         - TEST_DATA_VALIDATION_SIZE_ERROR: File size outside expected range
 *         - TEST_DATA_VALIDATION_INVALID_SPEC: File specification is malformed
 *
 * @note This function updates the validation report even on failure to provide diagnostic information
 * @note The function is thread-safe if different report structures are used for concurrent calls
 * @warning File specification must be validated before calling this function
 *
 * @see validate_file_spec_core() For file specification validation
 * @see test_data_file_info Structure definition for file specifications
 * @see test_data_validation_result Enumeration of possible validation results
 */
enum test_data_validation_result validate_test_file_core(const struct test_data_file_info* spec, 
                                                        struct test_data_validation_report* report);

/**
 * @brief Create a test file using the appropriate format-specific creator.
 *
 * This function orchestrates the creation of test data files by coordinating with
 * format-specific creation functions. It handles the complete file creation workflow
 * including directory preparation, format-specific content generation, validation
 * of created files, and proper error handling and reporting.
 *
 * Creation process workflow:
 * 1. **Specification Validation**: Verify the file specification is complete and valid
 * 2. **Directory Preparation**: Ensure the target directory exists and is writable
 * 3. **Conflict Resolution**: Handle existing files according to overwrite policies
 * 4. **Format Delegation**: Call appropriate format-specific creation function
 * 5. **Post-Creation Validation**: Verify the created file meets specifications
 * 6. **Permission Setting**: Apply appropriate file permissions for test usage
 * 7. **Report Updates**: Record creation results and update statistics
 *
 * Format-specific delegation:
 * - **GGUF files**: Delegates to `create_gguf_test_file()` for GGUF format generation
 * - **Parquet files**: Delegates to `create_parquet_test_file()` for Parquet format generation
 * - **Text files**: Delegates to `create_text_test_file()` for text content generation
 * - **Unknown formats**: Returns false with appropriate error logging
 *
 * File creation characteristics:
 * - **Minimal but valid**: Created files contain minimal data but are fully valid
 * - **Test-appropriate**: Content is suitable for testing scenarios and edge cases
 * - **Reproducible**: Same specifications produce identical files across runs
 * - **Efficient**: Files are small enough for fast test execution
 *
 * @param spec Pointer to the file specification defining the file to create.
 *             Must contain valid path, type, and size information. The specification
 *             determines the format, content characteristics, and validation criteria
 *             for the created file.
 * @param report Pointer to the validation report for tracking creation operations.
 *               Must not be NULL. The report is updated with creation results,
 *               including success/failure status, created file information, and
 *               any errors encountered during the creation process.
 * @return true if the file was created successfully and passes post-creation validation,
 *         false if any step in the creation process failed (directory creation,
 *         file generation, validation, or permission setting)
 *
 * @note Created files are automatically validated after creation to ensure correctness
 * @note The function will not overwrite existing valid files unless explicitly configured
 * @note Directory structure is created automatically if it doesn't exist
 * @warning Ensure sufficient disk space is available before calling this function
 *
 * @see validate_file_spec_core() For file specification validation
 * @see validate_test_file_core() For post-creation validation
 * @see test_data_file_info Structure definition for file specifications
 */
bool create_test_file_core(const struct test_data_file_info* spec, 
                          struct test_data_validation_report* report);

/**
 * @brief Fix file permissions to ensure proper test data accessibility.
 *
 * This function implements the core permission management algorithm that ensures
 * test data files have appropriate permissions for testing operations. It analyzes
 * current permissions, determines required changes, applies necessary modifications,
 * and verifies the changes were successful.
 *
 * Permission management algorithm:
 * 1. **Current State Analysis**: Read existing file permissions and ownership
 * 2. **Requirement Determination**: Calculate required permissions for test operations
 * 3. **Change Planning**: Determine minimal permission changes needed
 * 4. **Permission Application**: Apply necessary chmod operations
 * 5. **Verification**: Confirm permissions were changed successfully
 * 6. **Audit Logging**: Record permission changes for security audit trail
 *
 * Standard test file permissions:
 * - **Owner**: Read and write permissions for test data modification
 * - **Group**: Read permissions for shared test environments
 * - **Other**: Read permissions for general accessibility (configurable)
 * - **Execute**: Not granted for data files (security best practice)
 *
 * Platform-specific handling:
 * - **Unix/Linux**: Uses chmod() system call with appropriate mode bits
 * - **Windows**: Uses SetFileAttributes() with equivalent permission settings
 * - **macOS**: Handles extended attributes and ACLs when present
 *
 * @param path Path to the file whose permissions should be fixed. Must be a valid
 *             file path pointing to an existing file. The path can be relative or
 *             absolute, and the file must be accessible to the current process.
 * @return true if permissions were successfully analyzed and fixed (or were already
 *         correct), false if permission changes failed due to insufficient privileges,
 *         file system errors, or if the file does not exist
 *
 * @note This function requires appropriate privileges to modify file permissions
 * @note The function is idempotent - calling it multiple times has the same effect
 * @note Permission changes are logged for security auditing purposes
 * @warning This function may fail on read-only file systems or with insufficient privileges
 *
 * @see check_file_readable_core() For checking current file accessibility
 */
bool fix_file_permissions_core(const char* path);

/**
 * @brief Check if a file is readable using comprehensive accessibility analysis.
 *
 * This function implements the core file accessibility checking algorithm that
 * determines whether a file can be successfully read by the current process.
 * It performs comprehensive checks including existence, permissions, file system
 * status, and actual read capability verification.
 *
 * Accessibility checking algorithm:
 * 1. **Existence Verification**: Confirm the file exists at the specified path
 * 2. **Permission Analysis**: Check read permissions for current user/group/other
 * 3. **File System Status**: Verify file system is mounted and accessible
 * 4. **Lock Status Check**: Determine if file is locked by other processes
 * 5. **Actual Read Test**: Attempt to open file for reading (non-destructive)
 * 6. **Resource Availability**: Ensure sufficient file handles are available
 *
 * Platform-specific considerations:
 * - **Unix/Linux**: Uses access() system call and stat() for comprehensive checking
 * - **Windows**: Handles Windows-specific file attributes and security descriptors
 * - **Network filesystems**: Accounts for network latency and connectivity issues
 * - **Special filesystems**: Handles /proc, /sys, and other virtual filesystems appropriately
 *
 * Error conditions detected:
 * - File does not exist or path is invalid
 * - Insufficient read permissions for current process
 * - File is locked exclusively by another process
 * - File system errors (I/O errors, disk full, etc.)
 * - Network connectivity issues for remote files
 * - Resource exhaustion (too many open files)
 *
 * @param path Path to the file to check for readability. Can be relative or absolute.
 *             The path is resolved according to current working directory and
 *             environment settings. Symbolic links are followed automatically.
 * @return true if the file exists and can be successfully opened for reading by
 *         the current process, false if any accessibility check fails or if
 *         the file cannot be read for any reason
 *
 * @note This function performs actual file system operations and may be affected by I/O latency
 * @note The function follows symbolic links and checks the target file accessibility
 * @note Results may change between calls due to concurrent file system modifications
 * @warning Network file systems may introduce latency and temporary failures
 *
 * @see fix_file_permissions_core() For fixing permission issues
 */
bool check_file_readable_core(const char* path);

/**
 * @brief Check if a directory is accessible for read/write operations.
 *
 * Verifies that the specified directory exists and has appropriate permissions
 * for test data operations. This function checks both read and write access
 * to ensure full functionality.
 *
 * @param path Path to the directory to check
 * @return true if directory is accessible for read/write operations
 *
 * @note The function checks for both read and write permissions
 * @note Returns false if the directory doesn't exist
 * @note The function is safe to call multiple times
 */
bool check_directory_accessible_core(const char* path);

/**
 * @brief Create directory structure for test data files.
 *
 * Creates the necessary directory structure for test data files, including
 * parent directories if they don't exist. Sets appropriate permissions
 * for test data operations.
 *
 * @param path Path to the directory structure to create
 * @return true if directory structure was created successfully
 *
 * @note The function is idempotent - safe to call multiple times
 * @note Creates parent directories recursively as needed
 * @note Sets appropriate permissions for test data operations
 */
bool create_directory_structure_core(const char* path);

//
// Core error handling and utility functions
//

/**
 * @brief Get a human-readable string representation of a validation result.
 *
 * This utility function converts validation result codes into descriptive string
 * representations suitable for error messages, logging, and user interface display.
 * The returned strings are designed to be informative and actionable, helping
 * users understand validation failures and take appropriate corrective actions.
 *
 * String format characteristics:
 * - **Descriptive**: Clear explanation of what the result code means
 * - **Actionable**: Suggests potential solutions when appropriate
 * - **Consistent**: Uniform formatting and terminology across all result types
 * - **Localized**: Uses standard English terminology suitable for international use
 *
 * Supported result codes and their string representations:
 * - TEST_DATA_VALIDATION_SUCCESS: "Validation successful"
 * - TEST_DATA_VALIDATION_FILE_NOT_FOUND: "File not found"
 * - TEST_DATA_VALIDATION_PERMISSION_ERROR: "Permission denied"
 * - TEST_DATA_VALIDATION_FORMAT_ERROR: "Invalid file format"
 * - TEST_DATA_VALIDATION_CORRUPTION_ERROR: "File corruption detected"
 * - TEST_DATA_VALIDATION_SIZE_ERROR: "File size out of expected range"
 * - TEST_DATA_VALIDATION_INVALID_SPEC: "Invalid file specification"
 *
 * @param result Validation result code to convert to string representation.
 *               Should be one of the defined test_data_validation_result enumeration
 *               values. Unknown or invalid result codes will return a generic
 *               "Unknown validation result" message.
 * @return Constant string pointer to a human-readable description of the validation
 *         result. The returned string is statically allocated and remains valid
 *         for the lifetime of the program. The string should not be modified or freed.
 *
 * @note The returned string is constant and should not be modified by the caller
 * @note The function is thread-safe as it only accesses read-only static data
 * @note Unknown result codes are handled gracefully with a default message
 *
 * @see test_data_validation_result Enumeration of validation result codes
 */
const char* test_data_validation_result_to_string(enum test_data_validation_result result);

/**
 * @brief Validate a file specification structure for completeness and correctness.
 *
 * This function implements comprehensive validation of file specification structures
 * to ensure they contain all required information and that the information is
 * consistent and valid. It performs both structural validation (checking for
 * required fields) and semantic validation (ensuring values make sense).
 *
 * Validation checks performed:
 * 1. **Null Pointer Checks**: Verify the specification structure is not NULL
 * 2. **Path Validation**: Ensure file path is non-empty and syntactically valid
 * 3. **Type Validation**: Confirm dataset type is a recognized enumeration value
 * 4. **Size Constraints**: Verify size limits are reasonable and consistent
 * 5. **Format Consistency**: Check that file extension matches specified type
 * 6. **Range Validation**: Ensure numeric values are within acceptable ranges
 * 7. **String Validation**: Verify all string fields are properly null-terminated
 *
 * Specific validation criteria:
 * - **Path**: Must be non-NULL, non-empty, and contain valid path characters
 * - **Type**: Must be one of DATASET_TYPE_GGUF, DATASET_TYPE_PARQUET, DATASET_TYPE_TEXT
 * - **Expected size**: Must be positive and within reasonable limits (0 to MAX_FILE_SIZE)
 * - **Sequence constraints**: Must be positive integers if specified
 * - **File extension**: Should match the specified dataset type
 *
 * Common validation failures:
 * - NULL specification pointer
 * - Empty or NULL file path
 * - Unrecognized dataset type
 * - Negative or unreasonably large size constraints
 * - Path containing invalid characters for the target platform
 * - Inconsistent file extension and dataset type combination
 *
 * @param spec Pointer to the file specification structure to validate. Must not
 *             be NULL. All fields of the structure are examined for validity
 *             and consistency according to the validation criteria.
 * @return true if the specification passes all validation checks and can be
 *         safely used with other validation and creation functions, false if
 *         any validation check fails or if the specification is malformed
 *
 * @note This function should be called before using a file specification with other core functions
 * @note The function performs read-only validation and does not modify the specification
 * @note Validation is comprehensive but does not check file system accessibility
 * @warning Using an invalid specification with other functions may cause undefined behavior
 *
 * @see test_data_file_info Structure definition for file specifications
 * @see validate_test_file_core() For file validation using specifications
 * @see create_test_file_core() For file creation using specifications
 */
bool validate_file_spec_core(const struct test_data_file_info* spec);

/**
 * @brief Log a validation error using the centralized core logging system.
 *
 * This function provides standardized error logging for validation failures,
 * ensuring consistent error reporting across all validation operations. It
 * formats error messages with appropriate context information, timestamps,
 * and severity levels for effective debugging and monitoring.
 *
 * Logging functionality:
 * 1. **Error Formatting**: Creates structured error messages with context
 * 2. **Severity Classification**: Assigns appropriate log levels based on error type
 * 3. **Timestamp Addition**: Adds precise timestamps for error tracking
 * 4. **Context Enrichment**: Includes file path, error type, and system information
 * 5. **Output Routing**: Directs messages to appropriate logging destinations
 * 6. **Error Aggregation**: Supports error counting and pattern analysis
 *
 * Log message format:
 * ```
 * [TIMESTAMP] [LEVEL] Validation Error: [ERROR_TYPE] for file '[PATH]'
 * Details: [DETAILED_DESCRIPTION]
 * Suggestion: [RECOMMENDED_ACTION]
 * ```
 *
 * Severity level mapping:
 * - **ERROR**: Critical validation failures (corruption, format errors)
 * - **WARNING**: Non-critical issues (permission problems, size mismatches)
 * - **INFO**: Informational messages (successful validations, file creation)
 * - **DEBUG**: Detailed diagnostic information for troubleshooting
 *
 * Logging destinations:
 * - **Console**: Immediate feedback for interactive operations
 * - **Log files**: Persistent storage for batch operations and debugging
 * - **System logs**: Integration with system logging facilities (syslog, Event Log)
 * - **Structured logs**: JSON/XML format for automated processing
 *
 * @param path Path to the file that failed validation. Used for error context
 *             and to help identify problematic files. Can be relative or absolute
 *             path. NULL paths are handled gracefully with appropriate placeholder text.
 * @param result Validation result code indicating the type of failure that occurred.
 *               Used to determine error severity, generate appropriate error messages,
 *               and provide relevant troubleshooting suggestions.
 *
 * @note This function is thread-safe and can be called concurrently from multiple validation operations
 * @note Error messages are formatted for both human readability and automated processing
 * @note The function handles NULL path parameters gracefully
 * @warning High-frequency logging may impact performance during large-scale validation operations
 *
 * @see test_data_validation_result Enumeration of validation result codes
 * @see test_data_validation_result_to_string() For converting result codes to strings
 */
void log_validation_error_core(const char* path, enum test_data_validation_result result);

#ifdef __cplusplus
}
#endif
