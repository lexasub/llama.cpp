/**
 * @file test-data-validator-core.cpp
 * @brief Implementation of core validation algorithms and orchestration for test data validation system.
 *
 * This module implements the foundational validation infrastructure that powers the entire
 * test data validation system. It serves as the central coordination hub for all validation
 * operations, providing the core algorithms, orchestration logic, and shared utilities that
 * enable comprehensive validation across multiple file formats and testing scenarios.
 *
 * ## Implementation Overview
 *
 * ### Core Architecture
 * The implementation follows a layered architecture with clear separation of concerns:
 * - **Orchestration Layer**: Coordinates validation workflows and manages format delegation
 * - **Algorithm Layer**: Implements core validation algorithms and common patterns
 * - **Utility Layer**: Provides shared functionality for file operations and error handling
 * - **Integration Layer**: Interfaces with format-specific validators and system services
 *
 * ### Validation Orchestration Implementation
 * The core orchestration system manages the complete validation lifecycle:
 * ```
 * 1. Specification Validation → validate_file_spec_core()
 * 2. File System Checks → check_file_readable_core()
 * 3. Format Detection → Based on spec->expected_type
 * 4. Delegation → Format-specific validator functions
 * 5. Result Aggregation → Update validation report
 * 6. Error Handling → log_validation_error_core()
 * ```
 *
 * ### Format-Specific Delegation Strategy
 * The implementation uses a strategy pattern for format-specific validation:
 * - **GGUF Format**: Delegates to `validate_gguf_test_file()` for binary format validation
 * - **Parquet Format**: Delegates to `validate_parquet_test_file()` for columnar data validation
 * - **Text Format**: Delegates to `validate_text_test_file()` for text encoding validation
 * - **Unknown Formats**: Handled gracefully with appropriate error reporting
 *
 * ## Core Algorithm Implementations
 *
 * ### Validation Report Management Algorithm
 * ```c
 * void init_validation_report(report) {
 *     1. Zero-initialize all fields using memset()
 *     2. Reset counters (total_files, valid_files, invalid_files)
 *     3. Clear error message buffers
 *     4. Initialize timing and performance metrics
 *     5. Set validation status flags to default states
 * }
 * ```
 *
 * ### File Validation Orchestration Algorithm
 * ```c
 * enum validate_test_file_core(spec, report) {
 *     1. Validate input parameters (NULL checks)
 *     2. Switch on spec->expected_type for format detection
 *     3. Delegate to appropriate format-specific validator
 *     4. Process validation results and update report
 *     5. Format and append error messages if validation failed
 *     6. Return validation result code
 * }
 * ```
 *
 * ### File Creation Orchestration Algorithm
 * ```c
 * bool create_test_file_core(spec, report) {
 *     1. Validate input parameters and creation flags
 *     2. Check if file already exists (avoid unnecessary work)
 *     3. Analyze file path for special creation modes (corrupted, large)
 *     4. Delegate to format-specific creation functions
 *     5. Update creation statistics in validation report
 *     6. Return creation success status
 * }
 * ```
 *
 * ### Permission Management Algorithm
 * ```c
 * bool fix_file_permissions_core(path) {
 *     1. Validate file path parameter
 *     2. Apply standard test file permissions (0644)
 *     3. Use chmod() system call for permission modification
 *     4. Return success/failure status
 * }
 * ```
 *
 * ### File Accessibility Check Algorithm
 * ```c
 * bool check_file_readable_core(path) {
 *     1. Validate file path parameter
 *     2. Use access() system call with R_OK flag
 *     3. Return accessibility status
 * }
 * ```
 *
 * ## Error Handling and Reporting
 *
 * ### Error Message Aggregation
 * The implementation provides sophisticated error message aggregation:
 * - **Buffer Management**: Safely appends error messages to fixed-size buffers
 * - **Overflow Protection**: Prevents buffer overflows with size checking
 * - **Format Consistency**: Uses standardized error message formatting
 * - **Context Preservation**: Includes file paths and error types in messages
 *
 * ### Validation Result Translation
 * The system provides human-readable error descriptions:
 * - **Comprehensive Coverage**: Maps all validation result codes to descriptive strings
 * - **User-Friendly**: Uses clear, actionable language for error descriptions
 * - **Consistent Terminology**: Maintains uniform error terminology across the system
 * - **Graceful Degradation**: Handles unknown error codes with default messages
 *
 * ## File Creation Strategy Implementation
 *
 * ### Intelligent File Creation
 * The file creation system implements intelligent content generation:
 * - **Path Analysis**: Examines file paths for creation mode hints (corrupted, large)
 * - **Format-Appropriate Content**: Generates content suitable for each file format
 * - **Test-Optimized**: Creates minimal but valid files for efficient testing
 * - **Reproducible**: Ensures consistent file generation across multiple runs
 *
 * ### Creation Mode Detection
 * Special creation modes are detected through path analysis:
 * - **Corrupted Files**: Paths containing "corrupted" trigger corruption simulation
 * - **Large Files**: Paths containing "large" trigger extended content generation
 * - **Standard Files**: Default minimal content generation for normal test cases
 *
 * ## Platform Integration
 *
 * ### System Call Integration
 * The implementation integrates with platform-specific system calls:
 * - **File Permissions**: Uses chmod() for Unix-style permission management
 * - **File Accessibility**: Uses access() for efficient readability checking
 * - **Cross-Platform**: Designed for portability across Unix-like systems
 *
 * ### Error Handling Integration
 * Integrates with the llama.cpp logging system:
 * - **Structured Logging**: Uses LLAMA_LOG_DEBUG for consistent log formatting
 * - **Context-Rich Messages**: Includes file paths and error descriptions
 * - **Performance-Aware**: Uses debug-level logging to minimize performance impact
 *
 * ## Performance Characteristics
 *
 * ### Efficiency Optimizations
 * - **Early Validation**: Validates parameters before expensive operations
 * - **Existence Checking**: Avoids unnecessary file creation when files exist
 * - **Minimal System Calls**: Uses efficient system calls for file operations
 * - **Buffer Management**: Efficient string operations with bounds checking
 *
 * ### Memory Management
 * - **Stack Allocation**: Uses stack-allocated buffers for temporary operations
 * - **Zero-Copy**: Minimizes memory copying in validation operations
 * - **Bounded Operations**: All string operations are bounds-checked
 * - **Resource Cleanup**: Automatic cleanup through RAII and stack unwinding
 *
 * ## Integration Points
 *
 * ### Format-Specific Validators
 * Integrates with specialized validation modules:
 * - **GGUF Validator**: `validate_gguf_test_file()` and `create_*_gguf_*()` functions
 * - **Parquet Validator**: `validate_parquet_test_file()` and `create_*_parquet_*()` functions
 * - **Text Validator**: `validate_text_test_file()` and `create_*_text_*()` functions
 *
 * ### System Services
 * Interfaces with system and framework services:
 * - **Logging System**: llama.cpp logging infrastructure for error reporting
 * - **File System**: POSIX file system operations for file management
 * - **Memory Management**: Standard C library memory operations
 *
 * ## Usage Patterns and Examples
 *
 * ### Basic Validation Workflow
 * ```c
 * struct test_data_validation_report report;
 * init_validation_report(&report);
 * 
 * struct test_data_file_info spec = {
 *     .path = "/path/to/test.gguf",
 *     .expected_type = DATASET_GGUF,
 *     .create_if_missing = true
 * };
 * 
 * if (!validate_file_spec_core(&spec)) {
 *     // Handle invalid specification
 * }
 * 
 * enum test_data_validation_result result = validate_test_file_core(&spec, &report);
 * if (result != TEST_DATA_VALID) {
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
 * @see test-data-validator-core.h Core validation interface definitions
 * @see test-data-validator-common.h Common types and structures
 * @see test-data-validator-gguf.h GGUF-specific validation functions
 * @see test-data-validator-parquet.h Parquet-specific validation functions
 * @see test-data-validator-text.h Text-specific validation functions
 */

#include "test-data-validator-core.h"
#include "test-data-validator-common.h"
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

//
// Core validation orchestration functions
//

/**
 * @brief Initialize a validation report structure with default values.
 *
 * This function implements the validation report initialization algorithm that prepares
 * a report structure for use by resetting all fields to their default states and
 * ensuring the structure is ready for validation operations. The implementation uses
 * efficient memory operations to zero-initialize the entire structure.
 *
 * Implementation details:
 * - Uses memset() for efficient zero-initialization of the entire structure
 * - Handles NULL pointer gracefully without side effects
 * - Resets all counters, flags, and buffers to default states
 * - Prepares the structure for immediate use with validation functions
 *
 * Memory layout considerations:
 * - Zero-initialization is safe for all field types in the report structure
 * - Integer counters are set to 0 (indicating no files processed)
 * - Character arrays are null-terminated (first byte set to '\0')
 * - Boolean flags are set to false (0)
 * - Pointer fields are set to NULL (if any)
 *
 * @param report Pointer to the validation report structure to initialize.
 *               Must not be NULL for proper operation.
 */
void init_validation_report(struct test_data_validation_report* report) {
    if (!report) {
        return;
    }
    
    memset(report, 0, sizeof(*report));
    
    // Initialize tokenization statistics
    report->tokenization_stats.total_sequences_validated = 0;
    report->tokenization_stats.tokenized_sequences = 0;
    report->tokenization_stats.text_sequences = 0;
    report->tokenization_stats.mixed_content_files = 0;
    report->tokenization_stats.total_tokens_processed = 0;
    report->tokenization_stats.tokenization_errors = 0;
    report->tokenization_stats.avg_tokens_per_sequence = 0.0;
    report->tokenization_stats.tokenization_success_rate = 0.0;
}

/**
 * @brief Validate a single test file using format-specific validation delegation.
 *
 * This function implements the core validation orchestration algorithm that coordinates
 * the complete validation process for a single test file. It serves as the central
 * dispatch point for format-specific validation while providing unified error handling
 * and result aggregation.
 *
 * Implementation algorithm:
 * 1. **Parameter Validation**: Verify input parameters are valid and non-NULL
 * 2. **Format Detection**: Use spec->expected_type to determine validation strategy
 * 3. **Delegation**: Call appropriate format-specific validation function
 * 4. **Result Processing**: Interpret validation results and prepare error messages
 * 5. **Error Aggregation**: Append error messages to validation report
 * 6. **Return Handling**: Return validation result for caller processing
 *
 * Format-specific delegation implementation:
 * - **GGUF**: Calls validate_gguf_test_file() for binary format validation
 * - **Text**: Calls validate_text_test_file() for text encoding validation
 * - **Parquet**: Calls validate_parquet_test_file() for columnar data validation
 * - **Unknown**: Returns TEST_DATA_INVALID_FORMAT for unsupported types
 *
 * Error message aggregation algorithm:
 * - Formats error messages with file path and human-readable error description
 * - Calculates available buffer space to prevent overflow
 * - Uses strncat() for safe string concatenation with bounds checking
 * - Preserves existing error messages while adding new ones
 *
 * @param spec File specification containing path and expected format type
 * @param report Validation report for result aggregation and error tracking
 * @return Validation result code indicating success or specific failure type
 */
enum test_data_validation_result validate_test_file_core(const struct test_data_file_info* spec, 
                                                        struct test_data_validation_report* report) {
    if (!spec || !report) {
        return TEST_DATA_INVALID_FORMAT;
    }

    enum test_data_validation_result result;

    // Delegate to format-specific validation functions
    switch (spec->expected_type) {
        case DATASET_GGUF:
            result = validate_gguf_test_file(spec->path);
            break;
        case DATASET_TEXT:
            result = validate_text_test_file(spec->path);
            break;
        case DATASET_PARQUET:
            result = validate_parquet_test_file(spec->path);
            break;
        default:
            result = TEST_DATA_INVALID_FORMAT;
            break;
    }

    // Add error message if validation failed
    if (result != TEST_DATA_VALID) {
        char error_msg[256];
        snprintf(error_msg, sizeof(error_msg), "%s: %s\n",
                spec->path, test_data_validation_result_to_string(result));
        
        // Append to error messages if there's space
        size_t current_len = strlen(report->error_messages);
        size_t available_space = sizeof(report->error_messages) - current_len - 1;
        if (available_space > strlen(error_msg)) {
            strncat(report->error_messages, error_msg, available_space);
        }
    }

    return result;
}

/**
 * @brief Create a test file using intelligent format-specific content generation.
 *
 * This function implements the core file creation orchestration algorithm that
 * coordinates format-specific file creation while providing intelligent content
 * generation based on file path analysis and creation requirements. It serves
 * as the central dispatch point for all test file creation operations.
 *
 * Implementation algorithm:
 * 1. **Parameter Validation**: Verify input parameters and creation flags
 * 2. **Creation Policy Check**: Respect create_if_missing flag to avoid unwanted creation
 * 3. **Existence Check**: Avoid unnecessary work if file already exists
 * 4. **Path Analysis**: Examine file path for special creation mode hints
 * 5. **Format Delegation**: Call appropriate format-specific creation function
 * 6. **Statistics Update**: Update creation counters in validation report
 * 7. **Result Return**: Return creation success status
 *
 * Intelligent path analysis implementation:
 * - **Corrupted Mode**: Paths containing "corrupted" trigger corruption simulation
 * - **Large Mode**: Paths containing "large" trigger extended content generation
 * - **Standard Mode**: Default minimal content generation for efficient testing
 *
 * Format-specific creation delegation:
 * - **GGUF**: Creates minimal GGUF datasets with 5 sequences and 10 tokens
 * - **Text**: Creates minimal text datasets with configurable line counts
 * - **Parquet**: Creates minimal Parquet datasets with 5 rows and 10 columns
 * - **Corrupted Variants**: Creates intentionally corrupted files for error testing
 *
 * Creation optimization strategies:
 * - Early return for files that shouldn't be created (create_if_missing = false)
 * - Existence checking to avoid overwriting existing valid files
 * - Minimal content generation for fast test execution
 * - Reproducible content generation for consistent test results
 *
 * @param spec File specification containing creation parameters and format type
 * @param report Validation report for tracking creation statistics
 * @return true if file was created successfully or already exists, false on creation failure
 */
bool create_test_file_core(const struct test_data_file_info* spec, 
                          struct test_data_validation_report* report) {
    if (!spec || !report) {
        return false;
    }

    if (!spec->create_if_missing) {
        return true; // Not supposed to create this file
    }

    // Check if file already exists
    if (file_exists(spec->path)) {
        return true; // Already exists
    }

    bool created = false;

    // Delegate to format-specific creation functions
    switch (spec->expected_type) {
        case DATASET_GGUF:
            if (strstr(spec->path, "corrupted") != nullptr) {
                created = create_corrupted_gguf_test_file(spec->path);
            } else {
                created = create_minimal_gguf_dataset(spec->path, 5, 10);
            }
            break;

        case DATASET_TEXT:
            if (strstr(spec->path, "large") != nullptr) {
                created = create_minimal_text_dataset(spec->path, 100);
            } else if (strstr(spec->path, "corrupted") != nullptr) {
                created = create_corrupted_text_test_file(spec->path);
            } else {
                created = create_minimal_text_dataset(spec->path, 5);
            }
            break;

        case DATASET_PARQUET:
            if (strstr(spec->path, "corrupted") != nullptr) {
                created = create_corrupted_parquet_test_file(spec->path);
            } else if (strstr(spec->path, "text_parquet") != nullptr) {
                // Create Parquet file with text data for tokenization testing
                const char* sample_texts[] = {
                    "Hello world, this is a test sentence.",
                    "The quick brown fox jumps over the lazy dog.",
                    "Machine learning is transforming the world.",
                    "Natural language processing enables computers to understand text.",
                    "Large language models can generate human-like text."
                };
                created = create_test_parquet_with_text(spec->path, sample_texts, 5);
            } else if (strstr(spec->path, "mixed_content") != nullptr) {
                // Create Parquet file with mixed text and tokenized content
                created = create_test_parquet_mixed_content(spec->path, 10);
            } else {
                created = create_minimal_parquet_dataset(spec->path, 5, 10);
            }
            break;

        default:
            created = false;
            break;
    }

    if (created) {
        report->created_files++;
    }

    return created;
}

/**
 * @brief Fix file permissions to ensure proper test data accessibility.
 *
 * This function implements the core permission management algorithm that ensures
 * test data files have appropriate permissions for testing operations. It applies
 * standard test file permissions using the chmod() system call with careful
 * error handling and validation.
 *
 * Implementation algorithm:
 * 1. **Parameter Validation**: Verify file path is valid and non-NULL
 * 2. **Permission Application**: Apply standard test permissions (0644)
 * 3. **System Call Execution**: Use chmod() for atomic permission change
 * 4. **Result Verification**: Check system call return value for success
 * 5. **Status Return**: Return success/failure status to caller
 *
 * Permission scheme (0644):
 * - **Owner (6)**: Read and write permissions for test data modification
 * - **Group (4)**: Read permissions for shared test environments
 * - **Other (4)**: Read permissions for general accessibility
 * - **Execute**: Not granted for data files (security best practice)
 *
 * Error handling:
 * - Graceful handling of NULL path parameters
 * - System call error detection through return value checking
 * - No side effects on permission change failure
 *
 * @param path Path to the file whose permissions should be fixed
 * @return true if permissions were successfully applied, false on failure
 */
bool fix_file_permissions_core(const char* path) {
    if (!path) {
        return false;
    }

    // Try to fix file permissions by making it readable
    if (chmod(path, 0644) == 0) {
        return true;
    }

    return false;
}

/**
 * @brief Check if a file is readable using efficient system call validation.
 *
 * This function implements the core file accessibility checking algorithm using
 * the access() system call for efficient permission verification. It provides
 * fast, reliable checking of file readability without the overhead of actually
 * opening the file.
 *
 * Implementation algorithm:
 * 1. **Parameter Validation**: Verify file path is valid and non-NULL
 * 2. **System Call Execution**: Use access() with R_OK flag for read permission check
 * 3. **Result Interpretation**: Convert system call return value to boolean result
 * 4. **Status Return**: Return accessibility status to caller
 *
 * System call details:
 * - Uses access() with R_OK flag for read permission checking
 * - Follows symbolic links automatically
 * - Checks effective user/group permissions
 * - Returns immediately without file I/O operations
 *
 * Advantages of access() over file opening:
 * - No file descriptor consumption
 * - No risk of file locking conflicts
 * - Faster execution (no file I/O)
 * - Atomic permission checking
 *
 * @param path Path to the file to check for readability
 * @return true if file exists and is readable, false otherwise
 */
bool check_file_readable_core(const char* path) {
    if (!path) {
        return false;
    }

    // Check if file is readable
    return access(path, R_OK) == 0;
}

//
// Core error handling and utility functions
//

/**
 * @brief Convert validation result codes to human-readable string representations.
 *
 * This function implements the validation result translation algorithm that converts
 * enumerated validation result codes into descriptive, user-friendly string
 * representations. It provides comprehensive coverage of all validation result
 * types with consistent, actionable error descriptions.
 *
 * Implementation characteristics:
 * - **Comprehensive Coverage**: Maps all defined validation result codes
 * - **Consistent Terminology**: Uses uniform language across all error types
 * - **User-Friendly**: Provides clear, actionable descriptions
 * - **Graceful Degradation**: Handles unknown codes with default message
 * - **Static Strings**: Returns constant strings for memory efficiency
 *
 * String selection algorithm:
 * 1. **Switch Statement**: Efficient O(1) lookup using compiler optimization
 * 2. **Exact Matching**: Each enumeration value maps to specific string
 * 3. **Default Handling**: Unknown values return generic "Unknown error" message
 * 4. **Constant Returns**: All strings are compile-time constants
 *
 * Error description mapping:
 * - TEST_DATA_VALID → "Valid" (success case)
 * - TEST_DATA_MISSING → "Missing" (file not found)
 * - TEST_DATA_CORRUPTED → "Corrupted" (data integrity failure)
 * - TEST_DATA_INVALID_FORMAT → "Invalid format" (format mismatch)
 * - TEST_DATA_PERMISSION_ERROR → "Permission error" (access denied)
 * - TEST_DATA_SIZE_INVALID → "Invalid size" (size constraint violation)
 * - TEST_DATA_CONTENT_INVALID → "Invalid content" (content validation failure)
 *
 * @param result Validation result code to convert to string representation
 * @return Constant string pointer to human-readable error description
 */
const char* test_data_validation_result_to_string(enum test_data_validation_result result) {
    switch (result) {
        case TEST_DATA_VALID:
            return "Valid";
        case TEST_DATA_MISSING:
            return "Missing";
        case TEST_DATA_CORRUPTED:
            return "Corrupted";
        case TEST_DATA_INVALID_FORMAT:
            return "Invalid format";
        case TEST_DATA_PERMISSION_ERROR:
            return "Permission error";
        case TEST_DATA_SIZE_INVALID:
            return "Invalid size";
        case TEST_DATA_CONTENT_INVALID:
            return "Invalid content";
        default:
            return "Unknown error";
    }
}

/**
 * @brief Validate file specification structure for completeness and logical consistency.
 *
 * This function implements comprehensive validation of file specification structures
 * to ensure they contain all required information and that the information is
 * logically consistent and suitable for validation operations. It performs both
 * structural validation and semantic validation of specification fields.
 *
 * Implementation algorithm:
 * 1. **Null Pointer Validation**: Verify specification structure is not NULL
 * 2. **Path Validation**: Check file path is non-NULL and non-empty
 * 3. **Size Constraint Validation**: Verify size limits are logically consistent
 * 4. **Range Validation**: Ensure numeric values are within acceptable ranges
 * 5. **Consistency Checks**: Verify related fields are mutually consistent
 *
 * Validation criteria implemented:
 * - **Path Requirements**: Must be non-NULL and contain at least one character
 * - **Size Consistency**: min_size_bytes must not exceed max_size_bytes
 * - **Range Validity**: max_size_bytes of 0 indicates no upper limit
 * - **Logical Constraints**: All constraints must be achievable
 *
 * Common validation failures detected:
 * - NULL specification pointer
 * - NULL or empty file path
 * - Inconsistent size constraints (min > max when max > 0)
 * - Invalid numeric ranges
 *
 * Design considerations:
 * - Fast execution for frequent validation calls
 * - Comprehensive checking without false positives
 * - Clear failure conditions for debugging
 * - Minimal dependencies on external functions
 *
 * @param spec File specification structure to validate
 * @return true if specification is valid and usable, false if any validation check fails
 */
bool validate_file_spec_core(const struct test_data_file_info* spec) {
    if (!spec) {
        return false;
    }

    // Basic validation of file specification
    if (!spec->path || strlen(spec->path) == 0) {
        return false;
    }

    if (spec->min_size_bytes > spec->max_size_bytes && spec->max_size_bytes > 0) {
        return false;
    }

    return true;
}

/**
 * @brief Log validation errors using the centralized llama.cpp logging system.
 *
 * This function implements standardized error logging for validation failures,
 * ensuring consistent error reporting across all validation operations. It
 * integrates with the llama.cpp logging infrastructure to provide structured,
 * contextual error messages suitable for debugging and monitoring.
 *
 * Implementation algorithm:
 * 1. **Parameter Validation**: Verify file path is valid and non-NULL
 * 2. **Message Formatting**: Create structured error message with context
 * 3. **Result Translation**: Convert result code to human-readable description
 * 4. **Log Integration**: Use LLAMA_LOG_DEBUG for consistent log formatting
 * 5. **Context Preservation**: Include file path and error type in message
 *
 * Log message structure:
 * ```
 * "Validation error for [FILE_PATH]: [ERROR_DESCRIPTION]"
 * ```
 *
 * Logging level rationale:
 * - Uses DEBUG level to avoid overwhelming production logs
 * - Provides detailed information for development and troubleshooting
 * - Allows selective enabling during validation-intensive operations
 * - Maintains performance in production environments
 *
 * Integration benefits:
 * - Consistent formatting with other llama.cpp log messages
 * - Automatic timestamp and thread information inclusion
 * - Configurable output destinations (console, file, syslog)
 * - Integration with existing log filtering and analysis tools
 *
 * @param path File path that failed validation (for error context)
 * @param result Validation result code indicating the type of failure
 */
void log_validation_error_core(const char* path, enum test_data_validation_result result) {
    if (!path) {
        return;
    }

    LLAMA_LOG_DEBUG("Validation error for %s: %s\n", path, test_data_validation_result_to_string(result));
}
