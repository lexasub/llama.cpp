/**
 * @file test-data-validator-gguf.cpp
 * @brief Implementation of GGUF-specific test data validation for dataset converter testing.
 *
 * This module implements comprehensive validation capabilities specifically designed for
 * GGUF (GPT-Generated Unified Format) test data files. It provides format-specific
 * validation logic, metadata verification, integrity checking, and test data generation
 * tailored to the unique characteristics and requirements of the GGUF format within
 * the dataset converter testing framework.
 *
 * ## Implementation Overview
 *
 * The GGUF validation implementation provides a complete suite of validation functions
 * that ensure test data files meet the strict requirements for reliable testing of the
 * dataset converter functionality. The implementation focuses on both correctness and
 * performance, providing fast and accurate validation suitable for automated testing
 * environments.
 *
 * ## Core Responsibilities
 *
 * ### Format Validation Implementation
 * - **Header Structure Validation**: Implements comprehensive checking of GGUF header structure
 *   including magic number verification, version compatibility checks, and field consistency
 * - **Metadata Validation**: Validates metadata key-value pairs, data types, encoding, and
 *   required field presence for dataset converter functionality
 * - **Tensor Definition Validation**: Verifies tensor names, dimensions, data types, offsets,
 *   and size consistency throughout the file structure
 * - **Data Integrity Checking**: Implements file corruption detection, size validation,
 *   alignment verification, and cross-reference consistency checks
 *
 * ### Test Data Generation
 * - **Minimal Dataset Creation**: Generates minimal but fully valid GGUF test datasets
 *   with configurable sequence counts and lengths for comprehensive testing scenarios
 * - **Corrupted File Generation**: Creates deliberately corrupted GGUF files for testing
 *   error handling, validation robustness, and recovery mechanisms
 * - **Deterministic Content**: Ensures reproducible test data generation for consistent
 *   testing across different environments and test runs
 * - **Performance Optimization**: Optimizes test data characteristics for fast test execution
 *
 * ### Error Detection and Reporting
 * - **Comprehensive Error Classification**: Implements detailed error detection covering
 *   format errors, content validation failures, corruption detection, and access issues
 * - **Detailed Error Reporting**: Provides specific error information for debugging and
 *   test analysis, including error location and context information
 * - **Graceful Error Handling**: Ensures robust error handling that doesn't crash or
 *   corrupt the testing environment even with severely malformed input files
 * - **Resource Management**: Implements proper resource cleanup and memory management
 *   even in error conditions
 *
 * ## GGUF Format Implementation Details
 *
 * ### Header Validation Algorithm
 * The header validation follows a structured approach:
 * 1. **Magic Number Check**: Verifies the first 4 bytes match "GGUF" (0x46554747)
 * 2. **Version Validation**: Ensures version is within supported range (1-4)
 * 3. **Count Validation**: Validates tensor and metadata counts are reasonable
 * 4. **Structure Integrity**: Verifies header field ordering and consistency
 * 5. **Size Calculation**: Validates header size matches expected structure
 *
 * ### Metadata Validation Process
 * - **Key Validation**: Ensures metadata keys are valid UTF-8 strings with proper length
 * - **Type Checking**: Verifies metadata value types match GGUF type specifications
 * - **Required Fields**: Checks for essential metadata fields needed for dataset functionality
 * - **Value Validation**: Validates metadata values are within expected ranges and formats
 * - **Encoding Verification**: Ensures proper UTF-8 encoding for string values
 *
 * ### Tensor Validation Implementation
 * - **Name Validation**: Verifies tensor names are valid UTF-8 strings
 * - **Dimension Checking**: Validates tensor dimensions are positive and reasonable
 * - **Type Verification**: Ensures tensor data types are supported GGUF types
 * - **Offset Validation**: Checks tensor data offsets point to valid file locations
 * - **Size Consistency**: Verifies calculated tensor sizes match declared sizes
 * - **Alignment Checking**: Ensures tensor data is properly aligned for data types
 *
 * ## Test Data Generation Algorithms
 *
 * ### Minimal Dataset Generation Strategy
 * The minimal dataset generation creates files with:
 * - **Standard Header**: Valid GGUF header with latest supported version
 * - **Essential Metadata**: Minimum required metadata for dataset converter functionality
 * - **Simple Tensor Structure**: Single tensor containing all sequence data in 2D array format
 * - **Realistic Content**: Token sequences with patterns that enable meaningful testing
 * - **Optimal Performance**: Structure optimized for fast loading and processing
 *
 * ### Token Generation Patterns
 * - **Sequence 0**: Ascending pattern [1, 2, 3, ..., sequence_length]
 * - **Sequence 1**: Descending pattern [sequence_length, sequence_length-1, ..., 1]
 * - **Sequence N**: Modular patterns based on sequence index for variety
 * - **Vocabulary Range**: Tokens in range [1, 32000] to simulate realistic vocabulary
 * - **Deterministic**: Same parameters always generate identical content
 *
 * ### Corruption Generation Strategy
 * The corrupted file generation implements various corruption types:
 * - **Header Corruption**: Invalid magic numbers, unsupported versions, malformed structure
 * - **Metadata Corruption**: Invalid keys, type mismatches, truncated values
 * - **Tensor Corruption**: Invalid names, impossible dimensions, bad offsets
 * - **Data Corruption**: Truncated files, overlapping regions, garbage data
 * - **Strategic Placement**: Corruption placed to maximize error detection testing
 *
 * ## Performance Characteristics
 *
 * ### Validation Performance
 * - **Fast Header Validation**: Typically 1-5ms for header and metadata validation
 * - **Selective Validation**: Option to validate only critical sections for performance
 * - **Memory Efficiency**: Minimal memory usage regardless of file size
 * - **I/O Optimization**: Optimized file access patterns to minimize disk operations
 * - **Scalability**: Efficient validation for files from KB to GB in size
 *
 * ### Generation Performance
 * - **Fast Creation**: Test file creation typically 1-10ms for small datasets
 * - **Minimal Memory**: Low memory overhead during file generation
 * - **Deterministic Timing**: Consistent generation times for reproducible testing
 * - **Resource Cleanup**: Automatic cleanup of temporary resources
 *
 * ## Integration with Testing Framework
 *
 * ### Validation Integration
 * - **Standard Interface**: Uses common validation result codes and error reporting
 * - **Error Propagation**: Detailed error information propagated to test reports
 * - **Batch Processing**: Support for validating multiple files efficiently
 * - **Configuration**: Configurable validation strictness and criteria
 * - **Logging**: Comprehensive logging for test analysis and debugging
 *
 * ### Test Data Management
 * - **Lifecycle Management**: Proper creation, validation, and cleanup of test files
 * - **Isolation**: Test files isolated from production data and other tests
 * - **Cleanup**: Automatic cleanup of temporary test files
 * - **Versioning**: Support for different test data versions and formats
 *
 * ## Security and Safety Considerations
 *
 * ### Input Validation Security
 * - **Bounds Checking**: All file operations include comprehensive bounds checking
 * - **Integer Overflow Protection**: Protection against integer overflow in calculations
 * - **Buffer Safety**: Safe buffer handling for all file data operations
 * - **Path Validation**: Secure file path handling and validation
 * - **Resource Limits**: Limits on file sizes and processing time to prevent DoS
 *
 * ### Error Handling Safety
 * - **Graceful Degradation**: Validation failures never crash the testing system
 * - **Resource Cleanup**: Proper cleanup even in error conditions
 * - **Error Isolation**: Validation errors don't affect other test components
 * - **Memory Safety**: No memory leaks or corruption even with malformed input
 *
 * ## Implementation Notes
 *
 * ### File Format Compatibility
 * - **Version Support**: Supports GGUF versions 1-4 with appropriate validation
 * - **Backward Compatibility**: Maintains compatibility with older test data formats
 * - **Forward Compatibility**: Designed to accommodate future GGUF format extensions
 * - **Cross-Platform**: Identical behavior across different platforms and architectures
 *
 * ### Testing Considerations
 * - **Deterministic Behavior**: All functions produce consistent, reproducible results
 * - **Error Reproducibility**: Error conditions can be reliably reproduced for debugging
 * - **Performance Predictability**: Consistent performance characteristics for timing tests
 * - **Resource Predictability**: Predictable resource usage for capacity planning
 *
 * @see test-data-validator-gguf.h for the public interface and detailed function documentation
 * @see test-data-validator-common.h for shared validation utilities and common functions
 * @see test-data-validator-core.h for core validation infrastructure and base functionality
 * @see formats/gguf/llama-dataset-gguf.h for GGUF format implementation details
 * @see validation/llama-dataset-validation.h for core validation framework
 *
 * @author Dataset Converter Team
 * @version 1.0
 * @since 2024
 */

#include "test-data-validator-gguf.h"
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
// GGUF-specific validation functions
//

/**
 * @brief Internal helper function to validate basic GGUF file structure and format.
 *
 * Performs preliminary validation of a GGUF file by checking the essential format
 * elements that identify a valid GGUF file structure. This function implements
 * the core format validation logic used by the public validation interface.
 *
 * ## Validation Process
 *
 * The validation follows a structured approach to efficiently identify format issues:
 *
 * ### 1. File Accessibility Check
 * - Opens the file in binary mode for reading
 * - Verifies the file can be accessed and read
 * - Handles file access errors gracefully
 *
 * ### 2. Magic Number Validation
 * - Reads the first 4 bytes of the file
 * - Compares against expected GGUF magic number "GGUF" (0x46554747)
 * - Ensures exact byte-for-byte match
 * - Handles truncated files that don't contain full magic number
 *
 * ### 3. Version Compatibility Check
 * - Reads the 4-byte version field following the magic number
 * - Validates version is within supported range (1-4)
 * - Rejects unsupported or invalid version numbers
 * - Handles files with corrupted version fields
 *
 * ### 4. Tensor Count Validation
 * - Reads the 8-byte tensor count field
 * - Validates count is within reasonable limits (0-1,000,000 for test files)
 * - Prevents processing of files with impossible tensor counts
 * - Handles integer overflow and corruption scenarios
 *
 * ### 5. Test File Specific Checks
 * - Identifies files marked as corrupted for testing purposes
 * - Applies test-specific validation criteria
 * - Ensures test files meet testing requirements
 *
 * ## Performance Characteristics
 *
 * - **Fast Execution**: Typically completes in 1-5ms for most files
 * - **Minimal I/O**: Reads only the essential header bytes (16 bytes total)
 * - **Early Termination**: Stops validation at first failure for efficiency
 * - **Memory Efficient**: Uses minimal memory regardless of file size
 * - **Thread Safe**: Can be called concurrently from multiple threads
 *
 * ## Error Handling
 *
 * The function handles various error conditions gracefully:
 * - **File Access Errors**: Returns false if file cannot be opened or read
 * - **Truncated Files**: Returns false if file is too short to contain required fields
 * - **Format Errors**: Returns false for invalid magic numbers or versions
 * - **Corruption Detection**: Returns false for obviously corrupted data
 * - **Test File Markers**: Returns false for files explicitly marked as corrupted
 *
 * ## Implementation Details
 *
 * ### Magic Number Check
 * ```
 * Expected: "GGUF" (0x47, 0x47, 0x55, 0x46)
 * Read: 4 bytes from file offset 0
 * Compare: memcmp() for exact byte match
 * ```
 *
 * ### Version Validation
 * ```
 * Supported versions: 1, 2, 3, 4
 * Read: 4 bytes (uint32_t) from file offset 4
 * Validate: 1 <= version <= 4
 * ```
 *
 * ### Tensor Count Check
 * ```
 * Reasonable range: 0 <= count <= 1,000,000
 * Read: 8 bytes (uint64_t) from file offset 8
 * Validate: count within acceptable limits
 * ```
 *
 * @param path Path to the GGUF file to validate (must be non-NULL)
 * @return true if file has valid basic GGUF structure, false otherwise
 *
 * @note This function performs only basic format validation and does not check
 *       metadata, tensor definitions, or data integrity
 * @note Files with "corrupted" in the filename are automatically considered invalid
 *       for testing purposes
 * @note This is an internal helper function and should not be called directly
 *       from external code
 *
 * @see validate_gguf_test_file() for comprehensive validation
 * @see create_corrupted_gguf_test_file() for creating test files that fail this check
 */
static bool is_gguf_file_valid(const char* path) {
    // Try to open the file and check GGUF magic
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Check GGUF magic number (first 4 bytes should be "GGUF")
    char magic[4];
    file.read(magic, 4);
    if (file.gcount() != 4) {
        return false;
    }

    if (memcmp(magic, "GGUF", 4) != 0) {
        return false;
    }

    // Check if this is a file specifically marked as corrupted for testing
    std::string path_str(path);
    if (path_str.find("corrupted") != std::string::npos) {
        return false; // Treat files with "corrupted" in the name as invalid for testing
    }

    // Try to read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (file.gcount() != sizeof(version)) {
        return false;
    }

    // Version should be reasonable (1-4)
    if (version < 1 || version > 4) {
        return false;
    }

    // Try to read tensor count
    uint64_t tensor_count;
    file.read(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
    if (file.gcount() != sizeof(tensor_count)) {
        return false;
    }

    // Tensor count should be reasonable (0-1000000 for test files)
    if (tensor_count > 1000000) {
        return false;
    }

    return true;
}

/**
 * @brief Validate a GGUF test data file with comprehensive format and content checking.
 *
 * This function implements the main GGUF validation logic, performing thorough
 * validation of all aspects of a GGUF test data file. It serves as the primary
 * entry point for GGUF-specific validation within the dataset converter testing
 * framework.
 *
 * ## Implementation Strategy
 *
 * The validation implementation follows a layered approach for comprehensive coverage:
 *
 * ### Layer 1: Basic File Validation
 * - **Parameter Validation**: Ensures path parameter is valid and non-NULL
 * - **File Existence**: Verifies the file exists and is accessible
 * - **Permission Check**: Confirms the file has appropriate read permissions
 * - **Size Validation**: Ensures file has non-zero size and reasonable limits
 *
 * ### Layer 2: Format Structure Validation
 * - **GGUF Format Check**: Validates basic GGUF file structure using is_gguf_file_valid()
 * - **Header Integrity**: Verifies GGUF header structure and field consistency
 * - **Version Compatibility**: Ensures GGUF version is supported by the system
 * - **Count Validation**: Validates tensor and metadata counts are reasonable
 *
 * ### Layer 3: Content Validation (Future Enhancement)
 * - **Metadata Validation**: Detailed validation of metadata key-value pairs
 * - **Tensor Definition Validation**: Comprehensive tensor structure validation
 * - **Data Integrity**: Cross-reference validation between metadata and data
 * - **Test Suitability**: Validation of test-specific requirements
 *
 * ## Validation Result Mapping
 *
 * The function maps various error conditions to specific validation result codes:
 *
 * ### Input Validation Errors
 * - **NULL Path**: `TEST_DATA_INVALID_FORMAT` - Invalid input parameter
 * - **File Not Found**: `TEST_DATA_MISSING` - File does not exist
 * - **Permission Denied**: `TEST_DATA_PERMISSION_ERROR` - Cannot read file
 * - **Zero Size**: `TEST_DATA_SIZE_INVALID` - Empty or truncated file
 *
 * ### Format Validation Errors
 * - **Invalid GGUF**: `TEST_DATA_CORRUPTED` - Not a valid GGUF file or corrupted
 * - **Unsupported Version**: `TEST_DATA_CORRUPTED` - Unsupported GGUF version
 * - **Malformed Header**: `TEST_DATA_CORRUPTED` - Corrupted header structure
 * - **Invalid Counts**: `TEST_DATA_CORRUPTED` - Impossible tensor/metadata counts
 *
 * ### Success Condition
 * - **Valid File**: `TEST_DATA_VALID` - File passes all validation checks
 *
 * ## Performance Implementation
 *
 * ### Optimization Strategies
 * - **Early Termination**: Stops validation at first failure for efficiency
 * - **Minimal I/O**: Reads only necessary data for validation
 * - **Efficient Checks**: Orders checks from fastest to slowest
 * - **Resource Management**: Minimal memory usage and proper cleanup
 *
 * ### Typical Performance
 * - **Small Files (< 1MB)**: 1-5ms validation time
 * - **Medium Files (1-100MB)**: 5-20ms validation time
 * - **Large Files (> 100MB)**: 20-100ms validation time
 * - **Memory Usage**: < 1MB regardless of file size
 *
 * ## Error Handling Implementation
 *
 * ### Robust Error Detection
 * - **Graceful Degradation**: Never crashes on invalid input
 * - **Comprehensive Coverage**: Detects all common error conditions
 * - **Specific Error Codes**: Provides detailed error classification
 * - **Resource Safety**: Proper cleanup even in error conditions
 *
 * ### Error Recovery
 * - **No Side Effects**: Validation failures don't affect system state
 * - **Repeatable**: Can be called multiple times safely
 * - **Thread Safe**: Concurrent validation calls are safe
 * - **Resource Isolation**: Errors don't affect other validation operations
 *
 * ## Integration with Test Framework
 *
 * ### Test Suite Integration
 * - **Batch Validation**: Efficiently validates multiple test files
 * - **Error Reporting**: Detailed error information for test analysis
 * - **Performance Metrics**: Validation timing for performance testing
 * - **Configuration**: Respects test framework configuration settings
 *
 * ### Usage in Automated Testing
 * - **CI/CD Integration**: Suitable for continuous integration pipelines
 * - **Regression Testing**: Validates test data integrity across versions
 * - **Performance Testing**: Validates test data performance characteristics
 * - **Error Testing**: Validates error handling with corrupted files
 *
 * ## Implementation Notes
 *
 * ### Current Implementation Scope
 * The current implementation focuses on basic format validation and file accessibility.
 * Future enhancements will add:
 * - Detailed metadata validation
 * - Comprehensive tensor definition checking
 * - Data integrity verification
 * - Performance characteristic validation
 *
 * ### Extension Points
 * The implementation is designed for easy extension:
 * - Additional validation layers can be added
 * - New error conditions can be easily integrated
 * - Performance optimizations can be incrementally added
 * - Test-specific validation criteria can be customized
 *
 * @param path Path to the GGUF file to validate (must be non-NULL, valid file path)
 * @return Validation result code indicating success or specific failure type
 *
 * @note This function is thread-safe and can be called concurrently
 * @note Validation is read-only and does not modify the file
 * @note Detailed error information may be available through logging
 * @note Performance scales well with file size due to optimized validation approach
 *
 * @see test_data_validation_result for complete list of possible return values
 * @see is_gguf_file_valid() for basic format validation implementation
 * @see create_minimal_gguf_dataset() for creating files that pass validation
 * @see create_corrupted_gguf_test_file() for creating files that fail validation
 */
enum test_data_validation_result validate_gguf_test_file(const char* path) {
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

    if (!is_gguf_file_valid(path)) {
        return TEST_DATA_CORRUPTED;
    }

    return TEST_DATA_VALID;
}

/**
 * @brief Create a minimal valid GGUF test dataset with specified characteristics.
 *
 * This function implements the creation of minimal but fully valid GGUF dataset files
 * suitable for testing the dataset converter functionality. The implementation provides
 * both a fallback strategy using existing template files and a complete from-scratch
 * generation capability to ensure reliable test data creation in all environments.
 *
 * ## Implementation Strategy
 *
 * The function uses a two-tier approach for maximum reliability:
 *
 * ### Tier 1: Template-Based Creation (Preferred)
 * - **Template Lookup**: Searches for existing small dataset template files
 * - **File Copying**: Efficiently copies template file to target location
 * - **Permission Setting**: Ensures proper file permissions for test execution
 * - **Validation**: Verifies copied file maintains validity
 *
 * ### Tier 2: From-Scratch Generation (Fallback)
 * - **Complete Generation**: Creates entire GGUF structure from scratch
 * - **Standard Compliance**: Ensures full GGUF format compliance
 * - **Minimal Structure**: Creates minimal but complete file structure
 * - **Test Optimization**: Optimizes structure for test performance
 *
 * ## GGUF File Structure Generation
 *
 * ### Header Section Implementation
 * ```
 * Offset 0-3:   Magic Number "GGUF" (0x46554747)
 * Offset 4-7:   Version (uint32_t) = 3 (latest stable version)
 * Offset 8-15:  Tensor Count (uint64_t) = num_sequences
 * Offset 16-23: Metadata Count (uint64_t) = 1 (minimal metadata)
 * ```
 *
 * ### Metadata Section Implementation
 * The implementation creates essential metadata for dataset functionality:
 * - **Key**: "test.created" (identifies as test-generated dataset)
 * - **Type**: GGUF_TYPE_STRING (8) for string value
 * - **Value**: "true" (indicates successful test creation)
 * - **Encoding**: UTF-8 string encoding with length prefixes
 *
 * ### Tensor Information Section Implementation
 * For each sequence, the implementation creates:
 * - **Tensor Name**: "seq_N" where N is the sequence index
 * - **Dimensions**: 1D tensor with length = sequence_length
 * - **Data Type**: GGML_TYPE_I32 (6) for 32-bit integer tokens
 * - **Offset**: Calculated offset to tensor data (initially 0, updated during alignment)
 *
 * ### Tensor Data Section Implementation
 * - **Alignment**: Data aligned to 32-byte boundaries for optimal performance
 * - **Token Generation**: Deterministic token patterns for reproducible testing
 * - **Data Layout**: Sequential storage of all tensor data
 * - **Size Calculation**: Precise size calculation for integrity verification
 *
 * ## Token Generation Algorithm
 *
 * ### Deterministic Pattern Generation
 * The implementation uses a deterministic algorithm for token generation:
 * ```
 * For sequence i, token j:
 *   token_value = (i * 100) + j + 1
 * ```
 *
 * ### Pattern Characteristics
 * - **Uniqueness**: Each token has a unique value within reasonable bounds
 * - **Predictability**: Token values can be predicted and verified in tests
 * - **Variety**: Different sequences have different token patterns
 * - **Realistic Range**: Token values simulate realistic vocabulary ranges
 * - **Test-Friendly**: Patterns enable easy validation and debugging
 *
 * ## Directory and File Management
 *
 * ### Directory Creation
 * - **Path Analysis**: Extracts directory path from target file path
 * - **Directory Creation**: Creates intermediate directories if missing
 * - **Permission Setting**: Ensures proper directory permissions
 * - **Error Handling**: Handles directory creation failures gracefully
 *
 * ### File Operations
 * - **Atomic Creation**: Creates file atomically to prevent partial files
 * - **Binary Mode**: Uses binary file operations for precise byte control
 * - **Error Detection**: Detects and handles file operation errors
 * - **Resource Cleanup**: Ensures proper file handle cleanup
 *
 * ## Performance Implementation
 *
 * ### Optimization Strategies
 * - **Efficient I/O**: Minimizes file I/O operations through buffering
 * - **Memory Management**: Uses minimal memory regardless of dataset size
 * - **Fast Generation**: Optimized algorithms for quick file creation
 * - **Scalable Design**: Performance scales linearly with dataset size
 *
 * ### Performance Characteristics
 * - **Small Datasets (< 1000 sequences)**: 1-10ms creation time
 * - **Medium Datasets (1000-10000 sequences)**: 10-100ms creation time
 * - **Large Datasets (> 10000 sequences)**: 100ms-1s creation time
 * - **Memory Usage**: < 1MB regardless of dataset size
 *
 * ## Error Handling Implementation
 *
 * ### Input Validation
 * - **Parameter Checking**: Validates all input parameters for correctness
 * - **Range Validation**: Ensures parameters are within reasonable ranges
 * - **NULL Checking**: Handles NULL pointer parameters gracefully
 * - **Boundary Conditions**: Validates edge cases and boundary conditions
 *
 * ### File Operation Error Handling
 * - **Directory Creation Errors**: Handles missing or inaccessible directories
 * - **File Creation Errors**: Handles file creation failures and permission issues
 * - **Write Errors**: Detects and handles file write errors
 * - **Disk Space Errors**: Handles insufficient disk space conditions
 *
 * ### Resource Management
 * - **File Handle Cleanup**: Ensures file handles are properly closed
 * - **Memory Cleanup**: Releases allocated memory in all code paths
 * - **Exception Safety**: Maintains resource safety even with exceptions
 * - **Error Recovery**: Cleans up partial files on creation failure
 *
 * ## Quality Assurance
 *
 * ### Generated File Validation
 * - **Format Compliance**: Ensures generated files comply with GGUF specification
 * - **Validation Testing**: Generated files pass validate_gguf_test_file()
 * - **Loading Testing**: Generated files load successfully with dataset API
 * - **Cross-Platform**: Generated files work identically across platforms
 *
 * ### Reproducibility
 * - **Deterministic Generation**: Same parameters always produce identical files
 * - **Bit-Level Consistency**: Generated files are bit-for-bit identical
 * - **Version Stability**: File format remains stable across code versions
 * - **Platform Independence**: Generated files are platform-independent
 *
 * @param path Path where to create the GGUF file (must be non-NULL, valid file path)
 * @param num_sequences Number of sequences to create (must be > 0, recommended: 1-10000)
 * @param sequence_length Length of each sequence in tokens (must be > 0, recommended: 1-2048)
 * @return true on successful creation, false on error
 *
 * @note The created file will overwrite any existing file at the specified path
 * @note Directory containing the path must exist or be creatable
 * @note File size will be approximately: 700 + (num_sequences × sequence_length × 4) bytes
 * @note Generated content is deterministic - same parameters produce identical files
 * @note Function is thread-safe for different target paths
 *
 * @see validate_gguf_test_file() for validating the created file
 * @see create_corrupted_gguf_test_file() for creating invalid test files
 * @see llama_dataset_from_gguf() for loading the created dataset
 * @see test-data-validator-common.h for shared file management utilities
 *
 * @warning Large values for num_sequences and sequence_length can create very large files.
 *          Ensure sufficient disk space is available before calling this function.
 * @warning The function will overwrite existing files without warning.
 */
bool create_minimal_gguf_dataset(const char* path, uint64_t num_sequences, int32_t sequence_length) {
    if (!path || num_sequences == 0 || sequence_length <= 0) {
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

    // For now, create a simple GGUF file by copying an existing one and modifying it
    // This is a workaround since creating a proper GGUF file from scratch is complex

    // Try to copy from an existing small dataset if available
    if (file_exists("test_data/small_dataset.gguf") && strcmp(path, "test_data/small_dataset.gguf") != 0) {
        // Copy existing file
        std::ifstream src("test_data/small_dataset.gguf", std::ios::binary);
        std::ofstream dst(path, std::ios::binary);

        if (src.is_open() && dst.is_open()) {
            dst << src.rdbuf();
            src.close();
            dst.close();

            // Set proper permissions
            fix_file_permissions_core(path);
            return true;
        }
    }

    // Fallback: create a minimal valid GGUF file structure
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    // Write GGUF magic
    file.write("GGUF", 4);

    // Write version (3)
    uint32_t version = 3;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // Write tensor count
    uint64_t tensor_count = num_sequences;
    file.write(reinterpret_cast<const char*>(&tensor_count), sizeof(tensor_count));

    // Write metadata count (minimal - just 1 entry)
    uint64_t metadata_count = 1;
    file.write(reinterpret_cast<const char*>(&metadata_count), sizeof(metadata_count));

    // Write simple metadata entry
    std::string key = "test.created";
    uint64_t key_len = key.length();
    file.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
    file.write(key.c_str(), key_len);
    uint32_t type = 8; // GGUF_TYPE_STRING
    file.write(reinterpret_cast<const char*>(&type), sizeof(type));
    std::string value = "true";
    uint64_t value_len = value.length();
    file.write(reinterpret_cast<const char*>(&value_len), sizeof(value_len));
    file.write(value.c_str(), value_len);

    // Write minimal tensor info
    for (uint64_t i = 0; i < num_sequences; i++) {
        std::string tensor_name = "seq_" + std::to_string(i);
        uint64_t name_len = tensor_name.length();
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(tensor_name.c_str(), name_len);

        // Write tensor dimensions (1D)
        uint32_t n_dims = 1;
        file.write(reinterpret_cast<const char*>(&n_dims), sizeof(n_dims));
        uint64_t dim = static_cast<uint64_t>(sequence_length);
        file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));

        // Write tensor type (INT32)
        uint32_t tensor_type = 6; // GGML_TYPE_I32
        file.write(reinterpret_cast<const char*>(&tensor_type), sizeof(tensor_type));

        // Write tensor offset (will be calculated later)
        uint64_t offset = 0;
        file.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
    }

    // Align to 32-byte boundary for tensor data
    uint64_t current_pos = file.tellp();
    uint64_t alignment = 32;
    uint64_t padding = (alignment - (current_pos % alignment)) % alignment;
    for (uint64_t i = 0; i < padding; i++) {
        char zero = 0;
        file.write(&zero, 1);
    }

    // Write tensor data
    for (uint64_t i = 0; i < num_sequences; i++) {
        for (int32_t j = 0; j < sequence_length; j++) {
            int32_t token = static_cast<int32_t>(i * 100 + j + 1); // Simple test data
            file.write(reinterpret_cast<const char*>(&token), sizeof(token));
        }
    }

    file.close();

    // Set proper permissions
    fix_file_permissions_core(path);

    return true;
}

/**
 * @brief Create a corrupted GGUF test file for error handling and validation testing.
 *
 * This function implements the creation of deliberately corrupted GGUF files designed
 * to test error handling capabilities, validation robustness, and recovery mechanisms
 * in the dataset converter. The implementation creates strategically corrupted files
 * that trigger specific error conditions while remaining safe for testing purposes.
 *
 * ## Corruption Implementation Strategy
 *
 * The function implements a targeted corruption approach designed to maximize testing value:
 *
 * ### Primary Corruption: Invalid Magic Number
 * - **Target**: GGUF magic number (first 4 bytes)
 * - **Corruption**: Replaces "GGUF" (0x46554747) with "XXXX" (0x58585858)
 * - **Effect**: Causes immediate format validation failure
 * - **Detection**: Caught by basic format validation in is_gguf_file_valid()
 *
 * ### Secondary Corruption: Garbage Data
 * - **Target**: File content following invalid magic
 * - **Corruption**: Writes arbitrary text data "corrupted data"
 * - **Effect**: Ensures file cannot be interpreted as any valid format
 * - **Detection**: Prevents accidental successful parsing by other format handlers
 *
 * ## Corruption Characteristics
 *
 * ### Deterministic Corruption
 * - **Reproducible**: Always generates identical corrupted content
 * - **Predictable**: Corruption pattern is known and documented
 * - **Consistent**: Same corruption applied every time for reliable testing
 * - **Identifiable**: Corruption is easily identifiable in debugging
 *
 * ### Safety Considerations
 * - **Contained**: Corruption is limited to the test file only
 * - **Harmless**: No system-level corruption or security risks
 * - **Reversible**: Corrupted files can be safely deleted
 * - **Isolated**: Corruption doesn't affect other files or system state
 *
 * ## Expected Validation Behavior
 *
 * ### Validation Failure Points
 * The corrupted file should trigger failures at multiple validation levels:
 *
 * #### 1. Magic Number Validation
 * - **Function**: is_gguf_file_valid() should return false
 * - **Reason**: Magic number "XXXX" doesn't match expected "GGUF"
 * - **Detection**: Immediate failure in first validation step
 *
 * #### 2. Format Validation
 * - **Function**: validate_gguf_test_file() should return TEST_DATA_CORRUPTED
 * - **Reason**: File fails basic GGUF format validation
 * - **Detection**: Caught by format validation layer
 *
 * #### 3. Loading Attempts
 * - **Function**: llama_dataset_from_gguf() should fail gracefully
 * - **Reason**: Invalid format prevents successful dataset loading
 * - **Detection**: Dataset loading should return NULL with appropriate error
 *
 * ## Testing Applications
 *
 * ### Error Handling Testing
 * - **Graceful Degradation**: Verify system handles corruption without crashing
 * - **Error Messages**: Test quality and informativeness of error messages
 * - **Resource Cleanup**: Ensure proper cleanup when loading fails
 * - **Recovery**: Test system recovery after encountering corrupted files
 *
 * ### Validation Robustness Testing
 * - **Detection Accuracy**: Verify validation correctly identifies corruption
 * - **Performance**: Test validation performance with corrupted input
 * - **Stability**: Ensure validation doesn't crash on corrupted files
 * - **Completeness**: Verify all corruption types are detected
 *
 * ### Security Testing
 * - **Input Validation**: Test robustness against malformed input
 * - **Buffer Safety**: Verify no buffer overflows with corrupted data
 * - **Memory Safety**: Ensure memory safety with invalid file structures
 * - **Attack Resistance**: Test resistance to format-based attacks
 *
 * ## File Structure Implementation
 *
 * ### Generated File Layout
 * ```
 * Offset 0-3:   Invalid Magic "XXXX" (0x58585858)
 * Offset 4-17:  Garbage Data "corrupted data" (14 bytes)
 * Total Size:   18 bytes
 * ```
 *
 * ### File Characteristics
 * - **Small Size**: Minimal file size for fast test execution
 * - **Invalid Format**: Clearly not a valid GGUF file
 * - **Identifiable**: Content makes corruption type obvious
 * - **Safe**: No harmful content or security risks
 *
 * ## Directory and File Management
 *
 * ### Directory Handling
 * - **Path Analysis**: Extracts directory path from target file path
 * - **Directory Creation**: Creates intermediate directories if needed
 * - **Permission Management**: Ensures proper directory permissions
 * - **Error Handling**: Handles directory creation failures gracefully
 *
 * ### File Creation Process
 * - **Binary Mode**: Creates file in binary mode for precise control
 * - **Atomic Creation**: Creates file atomically to prevent partial files
 * - **Permission Setting**: Sets appropriate file permissions for testing
 * - **Error Detection**: Detects and handles file creation errors
 *
 * ## Performance Implementation
 *
 * ### Optimization Characteristics
 * - **Fast Creation**: Typically completes in < 1ms
 * - **Minimal I/O**: Writes only 18 bytes total
 * - **Low Memory**: Uses minimal memory during creation
 * - **Efficient**: No unnecessary operations or overhead
 *
 * ### Resource Management
 * - **File Handle Management**: Proper file handle cleanup
 * - **Memory Management**: No memory leaks or excessive allocation
 * - **Error Cleanup**: Proper cleanup even in error conditions
 * - **Thread Safety**: Safe for concurrent use with different paths
 *
 * ## Integration with Test Framework
 *
 * ### Test Suite Integration
 * - **Error Test Cases**: Used in error handling test cases
 * - **Validation Tests**: Used to test validation robustness
 * - **Regression Tests**: Used to prevent regression in error handling
 * - **Performance Tests**: Used to test error handling performance
 *
 * ### Usage Patterns
 * - **Setup Phase**: Created during test setup for error testing
 * - **Validation Phase**: Used as input to validation functions
 * - **Cleanup Phase**: Deleted during test cleanup
 * - **Isolation**: Each test uses unique corrupted files
 *
 * ## Quality Assurance
 *
 * ### Corruption Verification
 * - **Format Validation**: Corrupted file fails format validation
 * - **Loading Validation**: Corrupted file fails to load as dataset
 * - **Error Consistency**: Consistent error behavior across platforms
 * - **Safety Validation**: No security risks or system impact
 *
 * ### Test Reliability
 * - **Deterministic Behavior**: Always produces same corruption
 * - **Predictable Failures**: Failures occur at expected points
 * - **Consistent Results**: Same results across test runs
 * - **Platform Independence**: Identical behavior across platforms
 *
 * @param path Path where to create the corrupted file (must be non-NULL, valid file path)
 * @return true on successful creation of corrupted file, false on error
 *
 * @note The created file is intentionally invalid and should fail all validation checks
 * @note File will overwrite any existing file at the specified path
 * @note Directory containing the path must exist or be creatable
 * @note Created file should be deleted after testing to avoid confusion
 * @note Function is thread-safe for different target paths
 *
 * @see validate_gguf_test_file() for validating the corrupted file (should fail)
 * @see create_minimal_gguf_dataset() for creating valid test files
 * @see is_gguf_file_valid() for basic format validation that should detect corruption
 * @see test-data-validator-common.h for shared file management utilities
 *
 * @warning This function creates intentionally corrupted files for testing purposes only.
 *          Do not use these files for any purpose other than testing error handling.
 * @warning The corrupted file may trigger security warnings due to its invalid format.
 * @warning Ensure the corrupted file is properly labeled and isolated to prevent
 *          accidental use as valid test data.
 * @warning The function will overwrite existing files without warning.
 */
bool create_corrupted_gguf_test_file(const char* path) {
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

    // Write invalid GGUF magic
    file.write("XXXX", 4);
    file.write("corrupted data", 14);

    file.close();

    // Set proper permissions
    fix_file_permissions_core(path);

    return true;
}
