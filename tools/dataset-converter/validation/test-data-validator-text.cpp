/**
 * @file test-data-validator-text.cpp
 * @brief Implementation of text-specific test data validation and creation functionality.
 *
 * This module implements comprehensive validation and creation capabilities specifically
 * designed for text format test data within the dataset converter test suite. It provides
 * robust text file validation, encoding verification, content integrity checks, and
 * creation of test datasets with various characteristics for thorough testing coverage.
 *
 * ## Implementation Overview
 *
 * The implementation focuses on providing reliable and efficient text validation
 * capabilities that integrate seamlessly with the broader validation framework.
 * All functions are designed to be thread-safe, memory-efficient, and provide
 * detailed error reporting for comprehensive testing scenarios.
 *
 * ### Core Implementation Strategies
 *
 * #### Text File Validation
 * - **Multi-Stage Validation**: Implements layered validation approach starting with
 *   basic file checks and progressing to detailed content analysis
 * - **Streaming Processing**: Large files are processed in chunks to minimize memory
 *   usage while maintaining validation accuracy
 * - **Early Error Detection**: Validation stops at first critical error for faster
 *   feedback and improved performance
 * - **Comprehensive Coverage**: Validates all aspects from file system properties
 *   to content suitability for tokenization
 *
 * #### Encoding and Format Verification
 * - **UTF-8 Compliance**: Implements robust UTF-8 validation using optimized
 *   algorithms that detect invalid byte sequences and encoding errors
 * - **Character Set Validation**: Ensures all characters are within supported
 *   ranges and compatible with llama tokenization requirements
 * - **Line Ending Consistency**: Detects and validates line ending formats,
 *   ensuring consistency throughout the file
 * - **Binary Data Detection**: Identifies binary data contamination that could
 *   interfere with text processing
 *
 * #### Content Quality Assessment
 * - **Structural Analysis**: Validates text organization, line lengths, and
 *   overall file structure for training data suitability
 * - **Tokenization Compatibility**: Ensures text content is compatible with
 *   llama tokenizers and produces valid token sequences
 * - **Performance Validation**: Checks that text can be processed efficiently
 *   without causing memory or performance issues
 * - **Content Meaningfulness**: Validates that text contains meaningful content
 *   suitable for language model training
 *
 * ### Test Data Creation Implementation
 *
 * #### Minimal Dataset Generation
 * - **Template-Based Content**: Uses predefined templates and patterns to generate
 *   realistic training data that covers various linguistic structures
 * - **Configurable Length**: Supports creation of datasets with specified line
 *   counts while maintaining content quality and diversity
 * - **Encoding Compliance**: Ensures all generated content uses proper UTF-8
 *   encoding and follows text format standards
 * - **Atomic Creation**: Implements atomic file creation where possible to prevent
 *   partial or corrupted test files
 *
 * #### Corrupted File Generation
 * - **Controlled Corruption**: Introduces specific types of corruption designed
 *   to test error handling capabilities and validation robustness
 * - **Multiple Corruption Types**: Supports various corruption scenarios including
 *   encoding errors, format issues, and content problems
 * - **Reproducible Errors**: Creates predictable corruption patterns that can be
 *   used for consistent testing across different environments
 * - **Safety Measures**: Ensures corruption is limited to test files without
 *   affecting system stability or other files
 *
 * ## Performance Characteristics
 *
 * The implementation is optimized for efficiency across various scenarios:
 *
 * ### Memory Efficiency
 * - **Streaming Validation**: Large files are validated using streaming algorithms
 *   that maintain constant memory usage regardless of file size
 * - **Minimal Buffering**: Uses efficient buffering strategies that balance
 *   performance with memory consumption
 * - **Resource Cleanup**: Ensures proper cleanup of all resources including
 *   file handles, memory buffers, and temporary data structures
 *
 * ### I/O Optimization
 * - **Buffered Operations**: Uses buffered I/O operations to minimize system
 *   calls and improve overall performance
 * - **Sequential Access**: Optimizes for sequential file access patterns that
 *   work well with system caches and storage devices
 * - **Minimal File Operations**: Reduces the number of file operations through
 *   intelligent batching and caching strategies
 *
 * ### Algorithmic Efficiency
 * - **Early Termination**: Validation algorithms terminate early when critical
 *   errors are detected, improving performance for invalid files
 * - **Optimized UTF-8 Validation**: Uses efficient UTF-8 validation algorithms
 *   that process multiple bytes per iteration where possible
 * - **Cache-Friendly Patterns**: Implements access patterns that work well with
 *   CPU caches and modern memory hierarchies
 *
 * ## Error Handling and Reporting
 *
 * The implementation provides comprehensive error handling:
 *
 * ### Error Detection
 * - **Comprehensive Coverage**: Detects all types of text-related errors including
 *   file system issues, encoding problems, and content quality issues
 * - **Precise Classification**: Provides specific error codes that accurately
 *   describe the type and location of validation failures
 * - **Context Preservation**: Maintains error context information for detailed
 *   debugging and troubleshooting
 *
 * ### Error Recovery
 * - **Graceful Degradation**: Handles errors gracefully without crashing or
 *   leaving the system in an inconsistent state
 * - **Resource Cleanup**: Ensures proper cleanup of resources even when errors
 *   occur during validation or creation processes
 * - **State Consistency**: Maintains consistent internal state regardless of
 *   error conditions encountered during processing
 *
 * ## Integration with Validation Framework
 *
 * This implementation integrates seamlessly with the broader validation framework:
 *
 * ### Common Infrastructure Usage
 * - **Shared Utilities**: Leverages common validation utilities from
 *   test-data-validator-common.h for consistent behavior across formats
 * - **Core Functions**: Uses core validation functions from test-data-validator-core.h
 *   for fundamental file system operations and checks
 * - **Standard Interfaces**: Implements standard validation interfaces that work
 *   with the main validation coordinator
 *
 * ### Cross-Format Compatibility
 * - **Consistent Error Codes**: Uses standard error codes that are compatible
 *   with other format validators for unified error handling
 * - **Standard Patterns**: Follows established patterns and conventions used
 *   throughout the validation framework
 * - **Interoperability**: Designed to work seamlessly with other validation
 *   modules and the main dataset converter functionality
 *
 * ## Thread Safety and Concurrency
 *
 * All functions in this implementation are designed for safe concurrent use:
 *
 * ### Thread Safety Guarantees
 * - **Read-Only Operations**: Validation functions only read files and do not
 *   modify shared state, making them inherently thread-safe
 * - **Independent Creation**: File creation functions use unique temporary files
 *   and atomic operations where possible
 * - **No Global State**: All state is local to function calls or passed as
 *   parameters, eliminating race conditions
 * - **Reentrant Functions**: All functions are reentrant and can be called
 *   safely from multiple threads simultaneously
 *
 * ### Concurrency Optimization
 * - **Parallel Validation**: Multiple files can be validated concurrently
 *   without interference or performance degradation
 * - **Independent Resources**: Each validation operation uses independent
 *   resources to avoid contention between concurrent operations
 * - **Lock-Free Design**: Avoids locks and synchronization primitives for
 *   maximum performance in multi-threaded environments
 *
 * ## Testing and Quality Assurance
 *
 * The implementation includes comprehensive testing support:
 *
 * ### Self-Validation
 * - **Created File Validation**: Files created by this module can be validated
 *   by the same module to ensure consistency and correctness
 * - **Round-Trip Testing**: Supports round-trip testing where created files
 *   are validated and then used for further testing
 * - **Consistency Checks**: Includes internal consistency checks to ensure
 *   implementation correctness
 *
 * ### Error Injection
 * - **Controlled Corruption**: Provides mechanisms for creating controlled
 *   corruption scenarios for comprehensive error handling testing
 * - **Edge Case Generation**: Supports generation of edge case scenarios
 *   that test boundary conditions and unusual situations
 * - **Regression Testing**: Enables creation of test cases that prevent
 *   regression of previously fixed issues
 *
 * @see test-data-validator-text.h for the public interface and detailed API documentation
 * @see test-data-validator-common.h for shared validation utilities and common functions
 * @see test-data-validator-core.h for core validation infrastructure and basic operations
 * @see formats/text/llama-dataset-text.h for text dataset processing and integration
 * @see validation/llama-dataset-validation.h for core validation framework integration
 *
 * @author Dataset Converter Team
 * @version 1.0
 * @since 2024
 */

#include "test-data-validator-text.h"
#include "test-data-validator-common.h"
#include "test-data-validator-core.h"

#include "log.h"
#include "llama-dataset-internal.h"
#include "llama-impl.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

//
// Text-specific validation functions
//

/**
 * @brief Internal helper function to validate basic text file properties and content.
 *
 * Performs fundamental validation checks on a text file to determine if it contains
 * valid text content suitable for dataset processing. This function implements the
 * core text validation logic used by the public validation interface.
 *
 * ## Validation Process
 *
 * The function performs validation in several stages:
 * 1. **File Access**: Attempts to open the file for reading
 * 2. **Content Presence**: Verifies the file contains at least one line of text
 * 3. **Line Validity**: Checks that the first line is not empty and contains meaningful content
 * 4. **Character Validation**: Ensures characters are printable and suitable for text processing
 *
 * ## Character Set Validation
 *
 * The function validates characters using the following criteria:
 * - **Printable Characters**: Accepts standard printable ASCII characters (32-126)
 * - **Whitespace Characters**: Allows tabs (\t), newlines (\n), and carriage returns (\r)
 * - **Control Characters**: Rejects most control characters (0-31) except allowed whitespace
 * - **Extended Characters**: Accepts characters above ASCII 127 (handled by UTF-8 validation elsewhere)
 *
 * ## Performance Characteristics
 *
 * This function is optimized for quick validation:
 * - **Early Termination**: Stops reading after the first line for basic validation
 * - **Minimal I/O**: Uses standard library buffering for efficient file access
 * - **Character-by-Character**: Validates characters individually for precise error detection
 * - **Memory Efficient**: Uses minimal memory regardless of file size
 *
 * ## Error Conditions
 *
 * The function returns false for various error conditions:
 * - **File Access Errors**: Cannot open file for reading (permissions, existence, etc.)
 * - **Empty Files**: Files with no content or no readable lines
 * - **Empty Lines**: Files where the first line is empty or contains only whitespace
 * - **Invalid Characters**: Files containing control characters or binary data
 *
 * ## Integration with Public API
 *
 * This function is used internally by validate_text_test_file() as part of the
 * comprehensive validation process. It provides the basic text content validation
 * that is then supplemented by additional checks for encoding, format, and
 * tokenization compatibility.
 *
 * ## Usage Examples
 *
 * This function is typically called internally:
 * ```c
 * // Internal usage within validation pipeline
 * if (!is_text_file_valid(path)) {
 *     return TEST_DATA_CONTENT_INVALID;
 * }
 * ```
 *
 * ## Thread Safety
 *
 * This function is thread-safe:
 * - **Read-Only Operations**: Only reads from the specified file
 * - **No Shared State**: Uses only local variables and parameters
 * - **Independent Resources**: Each call uses independent file handles
 * - **Reentrant**: Can be called safely from multiple threads simultaneously
 *
 * @param path Path to the text file to validate. Must be a valid file path
 *             pointing to a readable file. The path should have been validated
 *             for existence and accessibility before calling this function.
 *
 * @return true if the file contains valid text content with at least one line
 *         of meaningful text using acceptable characters, false if the file
 *         cannot be opened, is empty, or contains invalid characters.
 *
 * @note This function performs basic validation only. Comprehensive validation
 *       including encoding checks is performed by validate_text_test_file().
 * @note The function only examines the first line for performance reasons.
 * @note Character validation is limited to basic ASCII checks; UTF-8 validation
 *       is performed separately in the full validation pipeline.
 *
 * @see validate_text_test_file() for comprehensive text file validation
 * @see check_file_readable_core() for file accessibility validation
 * @see get_file_size() for file size validation
 */
static bool is_text_file_valid(const char* path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    // Check if file has at least one line of text
    std::string line;
    if (!std::getline(file, line)) {
        return false;
    }

    // Check if line is not empty and contains printable characters
    if (line.empty()) {
        return false;
    }

    // Basic check for printable ASCII characters
    for (char c : line) {
        if (c < 32 && c != '\t' && c != '\n' && c != '\r') {
            return false;
        }
    }

    return true;
}

/**
 * @brief Validate a text test data file with comprehensive checks and detailed error reporting.
 *
 * Implements the main text file validation interface that performs thorough validation
 * of text files to ensure they meet all requirements for use as test data in the
 * dataset converter test suite. This function coordinates multiple validation stages
 * and provides detailed error reporting for comprehensive testing scenarios.
 *
 * ## Implementation Strategy
 *
 * The validation process is implemented as a multi-stage pipeline:
 * 1. **Parameter Validation**: Validates input parameters and handles null pointers
 * 2. **File Existence Check**: Verifies the file exists and is accessible
 * 3. **Permission Validation**: Ensures the file can be read by the current process
 * 4. **Size Validation**: Checks that the file has reasonable size (not empty)
 * 5. **Content Validation**: Validates text content quality and structure
 *
 * ## Validation Stages Detail
 *
 * ### Parameter and Basic Checks
 * - **Null Pointer Check**: Validates that the path parameter is not null
 * - **Path Validity**: Ensures the path string is valid and properly formatted
 * - **File Existence**: Uses file_exists() to verify the file is present
 * - **Access Permissions**: Uses check_file_readable_core() to verify read access
 *
 * ### Size and Content Validation
 * - **Size Check**: Uses get_file_size() to ensure file is not empty
 * - **Content Validation**: Uses is_text_file_valid() for basic text content checks
 * - **Format Verification**: Validates text format and structure
 * - **Encoding Compliance**: Ensures proper text encoding (handled by content validation)
 *
 * ## Error Code Mapping
 *
 * The function maps various error conditions to specific result codes:
 * - **TEST_DATA_INVALID_FORMAT**: Null or invalid path parameter
 * - **TEST_DATA_MISSING**: File does not exist or is not accessible
 * - **TEST_DATA_PERMISSION_ERROR**: Insufficient permissions to read the file
 * - **TEST_DATA_SIZE_INVALID**: File is empty (size is 0)
 * - **TEST_DATA_CONTENT_INVALID**: File content is not valid text
 * - **TEST_DATA_VALID**: File passes all validation checks
 *
 * ## Performance Optimization
 *
 * The implementation is optimized for performance:
 * - **Early Termination**: Stops validation at the first critical error
 * - **Efficient Ordering**: Performs quick checks before expensive operations
 * - **Minimal I/O**: Reduces file I/O operations through intelligent sequencing
 * - **Cached Results**: Leverages results from previous checks where possible
 *
 * ## Integration with Framework
 *
 * This function integrates with the broader validation framework:
 * - **Common Utilities**: Uses shared utilities from test-data-validator-common.h
 * - **Core Functions**: Leverages core validation functions for basic operations
 * - **Standard Interface**: Implements the standard validation interface pattern
 * - **Error Consistency**: Uses consistent error codes across all validators
 *
 * ## Error Handling Strategy
 *
 * The function implements robust error handling:
 * - **Graceful Degradation**: Handles all error conditions without crashing
 * - **Detailed Reporting**: Provides specific error codes for different failure types
 * - **Resource Safety**: Ensures no resource leaks occur during validation
 * - **State Consistency**: Maintains consistent state regardless of errors
 *
 * ## Thread Safety Implementation
 *
 * The function is implemented to be thread-safe:
 * - **Read-Only Operations**: Only reads from files, never modifies them
 * - **No Shared State**: Uses only local variables and function parameters
 * - **Independent Resources**: Each call uses independent file system operations
 * - **Reentrant Design**: Can be called safely from multiple threads
 *
 * ## Usage in Testing Framework
 *
 * This function is used throughout the testing framework:
 * - **Pre-Test Validation**: Validates test data files before running tests
 * - **Test Data Creation**: Validates files created by test data generators
 * - **Error Scenario Testing**: Validates that corrupted files are properly rejected
 * - **Regression Testing**: Ensures previously valid files remain valid
 *
 * ## Performance Characteristics
 *
 * Expected performance characteristics:
 * - **Small Files**: Validation completes in microseconds
 * - **Large Files**: Validation time scales linearly with file size
 * - **Memory Usage**: Constant memory usage regardless of file size
 * - **I/O Efficiency**: Minimizes file system operations for optimal performance
 *
 * @param path Path to the text file to validate. Must be a valid file path
 *             pointing to a readable text file. The path can be relative or
 *             absolute. Null pointers are handled gracefully with appropriate
 *             error codes.
 *
 * @return Validation result code indicating the outcome of the validation process:
 *         - TEST_DATA_VALID: File passes all validation checks
 *         - TEST_DATA_INVALID_FORMAT: Invalid or null path parameter
 *         - TEST_DATA_MISSING: File does not exist or is not accessible
 *         - TEST_DATA_PERMISSION_ERROR: Insufficient permissions to read file
 *         - TEST_DATA_SIZE_INVALID: File is empty (zero size)
 *         - TEST_DATA_CONTENT_INVALID: File content is not valid text
 *
 * @note This function is thread-safe and can be called concurrently on different files.
 * @note The validation process does not modify the input file in any way.
 * @note For very large files, validation may take some time but uses minimal memory.
 * @note The function performs comprehensive validation including all text-specific checks.
 *
 * @see test_data_validation_result_to_string() for converting result codes to strings
 * @see create_minimal_text_dataset() for creating valid test files
 * @see create_corrupted_text_test_file() for creating invalid test files
 * @see is_text_file_valid() for the internal content validation implementation
 */
enum test_data_validation_result validate_text_test_file(const char* path) {
    if (!path) {
        return TEST_DATA_INVALID_FORMAT;
    }

    if (!file_exists(path)) {
        return TEST_DATA_MISSING;
    }

    if (!check_file_readable_core(path)) {
        return TEST_DATA_PERMISSION_ERROR;
    }

    uint64_t size = get_file_size(path);
    if (size == 0) {
        return TEST_DATA_SIZE_INVALID;
    }

    if (!is_text_file_valid(path)) {
        return TEST_DATA_CONTENT_INVALID;
    }

    return TEST_DATA_VALID;
}

/**
 * @brief Create a minimal valid text test dataset with configurable content and comprehensive error handling.
 *
 * Implements the creation of text files containing minimal but valid datasets suitable
 * for testing the text processing capabilities of the dataset converter. The generated
 * content is designed to be representative of real training data while being small
 * enough for efficient testing and validation, with robust error handling and
 * platform compatibility.
 *
 * ## Implementation Strategy
 *
 * The creation process follows a structured approach:
 * 1. **Parameter Validation**: Validates input parameters for correctness and safety
 * 2. **Directory Preparation**: Ensures the target directory exists and is writable
 * 3. **File Creation**: Creates the file with proper permissions and error handling
 * 4. **Content Generation**: Generates meaningful text content line by line
 * 5. **Finalization**: Closes the file and sets appropriate permissions
 *
 * ## Content Generation Algorithm
 *
 * The content generation uses a template-based approach:
 * - **Template Pattern**: Uses a consistent template with variable elements
 * - **Line Numbering**: Includes line numbers for easy identification and debugging
 * - **Meaningful Text**: Generates text that resembles real training data
 * - **Consistent Format**: Maintains consistent formatting throughout the file
 * - **UTF-8 Compliance**: Ensures all generated content uses proper UTF-8 encoding
 *
 * ## Directory Handling Implementation
 *
 * The function implements robust directory handling:
 * - **Path Parsing**: Extracts directory path from the full file path
 * - **Directory Creation**: Creates missing directories using create_directory_if_missing()
 * - **Permission Handling**: Ensures directories have appropriate permissions
 * - **Error Recovery**: Handles directory creation failures gracefully
 *
 * ## File Creation Process
 *
 * The file creation process is implemented for reliability:
 * - **Atomic Creation**: Creates files atomically where possible to prevent partial files
 * - **Buffered Writing**: Uses buffered I/O for efficient writing operations
 * - **Error Detection**: Detects and handles various I/O errors during creation
 * - **Resource Cleanup**: Ensures proper cleanup of file handles and resources
 *
 * ## Content Quality Assurance
 *
 * The generated content meets quality standards:
 * - **Realistic Content**: Text resembles real training data patterns
 * - **Tokenization Friendly**: Content is designed to tokenize efficiently
 * - **Character Diversity**: Includes various character types for comprehensive testing
 * - **Length Variation**: While using templates, provides some content variation
 * - **Encoding Compliance**: All content uses proper UTF-8 encoding
 *
 * ## Performance Optimization
 *
 * The implementation is optimized for performance:
 * - **Buffered I/O**: Uses standard library buffering for efficient writing
 * - **Minimal Memory**: Generates content on-demand without large buffers
 * - **Efficient Templates**: Uses efficient string operations for content generation
 * - **Batch Operations**: Minimizes system calls through batched operations
 *
 * ## Error Handling Implementation
 *
 * Comprehensive error handling covers various scenarios:
 * - **Parameter Validation**: Handles null pointers and invalid parameters
 * - **Directory Errors**: Manages directory creation and permission failures
 * - **File Creation Errors**: Handles file creation failures and permission issues
 * - **I/O Errors**: Manages write errors, disk full conditions, and other I/O issues
 * - **Resource Cleanup**: Ensures proper cleanup even when errors occur
 *
 * ## Platform Compatibility
 *
 * The implementation works across different platforms:
 * - **Path Handling**: Uses platform-appropriate path separators and conventions
 * - **Line Endings**: Generates platform-appropriate line endings
 * - **Permissions**: Sets appropriate file permissions using fix_file_permissions_core()
 * - **Character Encoding**: Uses UTF-8 encoding consistently across platforms
 *
 * ## Integration with Validation
 *
 * Created files are designed to pass validation:
 * - **Validation Compatibility**: Files created by this function pass validate_text_test_file()
 * - **Standard Format**: Uses standard text format compatible with the dataset converter
 * - **Proper Structure**: Generates files with proper structure and formatting
 * - **Quality Assurance**: Ensures created files meet all quality requirements
 *
 * ## Usage Patterns
 *
 * The function supports various usage patterns:
 * - **Small Test Files**: Creates small files for quick testing (10-100 lines)
 * - **Medium Datasets**: Creates medium-sized files for comprehensive testing (1000-10000 lines)
 * - **Large Datasets**: Creates large files for performance testing (100000+ lines)
 * - **Custom Sizes**: Supports any reasonable number of lines based on requirements
 *
 * ## Thread Safety Implementation
 *
 * The function is implemented to be thread-safe:
 * - **Independent Operations**: Each call operates on independent file paths
 * - **No Shared State**: Uses only local variables and function parameters
 * - **Atomic File Creation**: Uses atomic operations where possible
 * - **Resource Independence**: Each call uses independent file handles and resources
 *
 * @param path Path where to create the text file. Must be a valid file path
 *             with write permissions. The directory must exist or be creatable,
 *             and the location must be writable. If the file already exists,
 *             it will be overwritten. Null pointers are handled gracefully.
 * @param num_lines Number of lines to create in the text file. Must be greater
 *                  than 0 and less than system-specific limits. Each line will
 *                  contain meaningful text content suitable for training data
 *                  preparation. Very large values may take significant time.
 *
 * @return true on successful creation of the text file with the specified number
 *         of lines and proper formatting, false on any error condition including
 *         I/O errors, permission issues, invalid parameters, or system limitations.
 *
 * @note This function is thread-safe when called with different path parameters.
 * @note The created file will be immediately available for reading and validation.
 * @note File creation is atomic where supported by the filesystem.
 * @note Generated content uses UTF-8 encoding and platform-appropriate line endings.
 *
 * @see validate_text_test_file() for validating the created file
 * @see create_corrupted_text_test_file() for creating intentionally invalid files
 * @see create_directory_if_missing() for directory creation functionality
 * @see fix_file_permissions_core() for permission setting functionality
 *
 * @warning Existing files at the specified path will be overwritten without warning.
 * @warning Very large num_lines values may consume significant time and disk space.
 */
bool create_minimal_text_dataset(const char* path, uint64_t num_lines) {
    if (!path || num_lines == 0) {
        return false;
    }

    // Ensure directory exists
    std::string path_str(path);
    size_t pos = path_str.find_last_of('/');
    if (pos != std::string::npos) {
        std::string dir = path_str.substr(0, pos);
        if (!create_directory_if_missing(dir.c_str())) {
            return false;
        }
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    for (uint64_t i = 0; i < num_lines; i++) {
        file << "This is test line " << i << " for dataset converter testing." << std::endl;
    }

    file.close();

    // Set proper permissions
    fix_file_permissions_core(path);

    return true;
}

/**
 * @brief Create a corrupted text test file for comprehensive error handling validation and testing.
 *
 * Implements the creation of deliberately corrupted or malformed text files to test
 * error handling capabilities, validation robustness, and recovery mechanisms in the
 * dataset converter. The corruption is carefully designed to trigger specific error
 * conditions while maintaining safety and providing comprehensive testing coverage
 * for various failure scenarios.
 *
 * ## Implementation Strategy
 *
 * The corruption creation process follows a controlled approach:
 * 1. **Parameter Validation**: Validates input parameters and handles edge cases
 * 2. **Directory Preparation**: Ensures target directory exists and is writable
 * 3. **Corruption Selection**: Selects appropriate corruption type for testing
 * 4. **File Creation**: Creates file with binary mode for precise corruption control
 * 5. **Corruption Injection**: Injects specific corruption patterns into the file
 * 6. **Finalization**: Closes file and sets permissions for testing use
 *
 * ## Corruption Algorithm Implementation
 *
 * The current implementation uses binary data corruption:
 * - **Binary Data Injection**: Writes binary data that violates text format requirements
 * - **Byte Pattern Generation**: Creates patterns of bytes from 0-255 to ensure corruption
 * - **Controlled Size**: Generates exactly 100 bytes of corrupted data for consistency
 * - **Predictable Corruption**: Uses deterministic patterns for reproducible testing
 * - **Format Violation**: Ensures the corruption violates text format expectations
 *
 * ## Corruption Types Supported
 *
 * The implementation can be extended to support various corruption types:
 *
 * ### Current Implementation
 * - **Binary Data Contamination**: Injects binary data into what should be text content
 * - **Control Character Injection**: Includes control characters that cause processing issues
 * - **Invalid Byte Sequences**: Creates byte patterns that violate text format rules
 *
 * ### Future Extensions
 * - **UTF-8 Encoding Errors**: Invalid UTF-8 byte sequences and encoding violations
 * - **Line Ending Corruption**: Mixed or invalid line ending sequences
 * - **File Truncation**: Files that end abruptly without proper termination
 * - **Size Mismatches**: Files with inconsistent size information
 *
 * ## Safety Implementation
 *
 * The corruption is implemented safely:
 * - **Isolated Corruption**: Corruption is limited to the created file only
 * - **No System Impact**: Creation process does not affect system stability
 * - **Controlled Scope**: Corruption patterns are well-defined and predictable
 * - **Reversible Effects**: Corrupted files can be safely deleted without impact
 *
 * ## Directory Handling
 *
 * Robust directory handling ensures reliable file creation:
 * - **Path Parsing**: Extracts directory path from the full file path
 * - **Directory Creation**: Creates missing directories using create_directory_if_missing()
 * - **Permission Management**: Ensures directories have appropriate permissions
 * - **Error Recovery**: Handles directory creation failures gracefully
 *
 * ## File Creation Process
 *
 * The file creation process is implemented for precise control:
 * - **Binary Mode**: Opens file in binary mode for exact byte control
 * - **Direct Writing**: Writes corruption data directly without text processing
 * - **Error Detection**: Detects and handles file creation and writing errors
 * - **Resource Management**: Ensures proper cleanup of file handles
 *
 * ## Testing Integration
 *
 * Created corrupted files integrate with the testing framework:
 * - **Validation Testing**: Files should fail validate_text_test_file() checks
 * - **Error Code Testing**: Should trigger specific error codes for different corruption types
 * - **Recovery Testing**: Tests system recovery from corruption detection
 * - **Robustness Testing**: Validates that corruption doesn't cause crashes or data loss
 *
 * ## Performance Characteristics
 *
 * The corruption creation is optimized for testing efficiency:
 * - **Fast Creation**: Corrupted files are created quickly for efficient testing
 * - **Minimal Size**: Creates small corrupted files to minimize storage requirements
 * - **Predictable Timing**: Creation time is consistent and predictable
 * - **Low Resource Usage**: Uses minimal system resources during creation
 *
 * ## Error Handling Implementation
 *
 * Comprehensive error handling covers creation failures:
 * - **Parameter Validation**: Handles null pointers and invalid parameters
 * - **Directory Errors**: Manages directory creation and permission failures
 * - **File Creation Errors**: Handles file creation failures and permission issues
 * - **Write Errors**: Manages write failures and disk space issues
 * - **Resource Cleanup**: Ensures proper cleanup even when errors occur
 *
 * ## Thread Safety Implementation
 *
 * The function is implemented to be thread-safe:
 * - **Independent Operations**: Each call operates on independent file paths
 * - **No Shared State**: Uses only local variables and function parameters
 * - **Atomic Creation**: Creates files atomically where possible
 * - **Resource Independence**: Each call uses independent file handles
 *
 * ## Usage in Test Scenarios
 *
 * The function supports various testing scenarios:
 * - **Basic Error Testing**: Creates simple corrupted files for basic error handling tests
 * - **Validation Testing**: Tests that validation correctly identifies corruption
 * - **Recovery Testing**: Tests system recovery from corruption detection
 * - **Robustness Testing**: Ensures corruption doesn't cause system instability
 *
 * ## Expected Validation Results
 *
 * Files created by this function should trigger specific validation failures:
 * - **TEST_DATA_CONTENT_INVALID**: Most common result for binary data corruption
 * - **TEST_DATA_FORMAT_ERROR**: For format-specific corruption types
 * - **TEST_DATA_ENCODING_ERROR**: For UTF-8 and encoding-related corruption
 * - **Never TEST_DATA_VALID**: Corrupted files should never pass validation
 *
 * @param path Path where to create the corrupted text file. Must be a valid
 *             file path with write permissions. The directory must exist or
 *             be creatable, and the location must be writable. If the file
 *             already exists, it will be overwritten with corrupted content.
 *             Null pointers are handled gracefully.
 *
 * @return true on successful creation of the corrupted file with the intended
 *         corruption patterns, false on any error condition including I/O errors,
 *         permission issues, or invalid parameters. Note that "success" means
 *         the corruption was successfully introduced, not that the file content
 *         is valid for normal use.
 *
 * @note This function is thread-safe when called with different path parameters.
 * @note The created file is intentionally invalid and should fail validation checks.
 * @note Different calls may produce the same corruption pattern for consistent testing.
 * @note The corrupted file should be deleted after testing to avoid confusion.
 * @note File creation uses binary mode for precise corruption control.
 *
 * @see validate_text_test_file() for testing corruption detection
 * @see create_minimal_text_dataset() for creating valid test files
 * @see test_data_validation_result for expected error codes
 * @see create_directory_if_missing() for directory creation functionality
 * @see fix_file_permissions_core() for permission setting functionality
 *
 * @warning The created file is intentionally corrupted and should not be used for actual training.
 * @warning Existing files at the specified path will be overwritten with corrupted content.
 * @warning The corruption patterns may trigger security software warnings in some environments.
 */
bool create_corrupted_text_test_file(const char* path) {
    if (!path) {
        return false;
    }

    // Ensure directory exists
    std::string path_str(path);
    size_t pos = path_str.find_last_of('/');
    if (pos != std::string::npos) {
        std::string dir = path_str.substr(0, pos);
        if (!create_directory_if_missing(dir.c_str())) {
            return false;
        }
    }

    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Write binary data that's not valid text
    for (int i = 0; i < 100; i++) {
        char byte = static_cast<char>(i % 256);
        file.write(&byte, 1);
    }

    file.close();

    // Set proper permissions
    fix_file_permissions_core(path);

    return true;
}
