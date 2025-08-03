#pragma once

/**
 * @file test-data-validator-gguf.h
 * @brief GGUF-specific test data validation module for dataset converter testing.
 *
 * This module provides comprehensive validation capabilities specifically designed for
 * GGUF (GPT-Generated Unified Format) test data files. It implements format-specific
 * validation logic, metadata verification, and integrity checking tailored to the
 * unique characteristics and requirements of the GGUF format.
 *
 * ## GGUF Format Validation Overview
 *
 * GGUF is a binary format with specific structural requirements that must be validated
 * to ensure test data integrity and compatibility with the dataset converter framework.
 * This module provides specialized validation for all aspects of the GGUF format.
 *
 * ## Key Responsibilities
 *
 * ### Format Structure Validation
 * - **Header Validation**: Verifies GGUF magic number, version compatibility, and header structure
 * - **Metadata Section**: Validates metadata key-value pairs, data types, and encoding
 * - **Tensor Information**: Verifies tensor definitions, dimensions, and data type specifications
 * - **Data Section**: Validates tensor data integrity, alignment, and size consistency
 * - **File Structure**: Ensures proper section ordering and offset calculations
 *
 * ### Metadata Verification
 * - **Standard Keys**: Validates presence and format of required training.* metadata keys
 * - **Data Types**: Verifies metadata value types match their declared types
 * - **Encoding**: Validates string encoding (UTF-8) and binary data integrity
 * - **Consistency**: Ensures metadata values are consistent with tensor data
 * - **Completeness**: Checks for required metadata fields for dataset functionality
 *
 * ### Integrity Checking
 * - **File Corruption**: Detects truncated files, invalid offsets, and data corruption
 * - **Checksum Validation**: Verifies data integrity using built-in checksums where available
 * - **Cross-Reference**: Validates consistency between metadata and actual tensor data
 * - **Size Validation**: Ensures file size matches expected size based on metadata
 * - **Alignment**: Verifies proper data alignment for optimal performance
 *
 * ### Test Data Specific Validation
 * - **Minimal Requirements**: Ensures test files meet minimum requirements for testing
 * - **Content Validation**: Validates that test data contains appropriate content for testing
 * - **Performance Characteristics**: Verifies test files have expected performance characteristics
 * - **Compatibility**: Ensures test files are compatible with all dataset converter features
 * - **Error Scenarios**: Validates that corrupted test files trigger appropriate error handling
 *
 * ## GGUF Validation Criteria
 *
 * ### Header Validation Criteria
 * - **Magic Number**: Must be exactly "GGUF" (0x46554747)
 * - **Version**: Must be a supported GGUF version (currently 1, 2, or 3)
 * - **Tensor Count**: Must be non-negative and consistent with file content
 * - **Metadata Count**: Must be non-negative and match actual metadata entries
 * - **Header Size**: Must be consistent with declared counts and data
 *
 * ### Metadata Validation Criteria
 * - **Key Format**: Keys must be valid UTF-8 strings with appropriate naming conventions
 * - **Value Types**: Values must match their declared GGUF data types
 * - **Required Fields**: Must contain essential metadata for dataset functionality
 * - **Optional Fields**: Optional metadata must be properly formatted if present
 * - **Size Limits**: Metadata size must be within reasonable limits for test data
 *
 * ### Tensor Validation Criteria
 * - **Tensor Names**: Must be valid UTF-8 strings following naming conventions
 * - **Dimensions**: Must have valid dimension counts and sizes
 * - **Data Types**: Must use supported GGUF tensor data types
 * - **Offsets**: Tensor data offsets must be valid and non-overlapping
 * - **Sizes**: Tensor data sizes must match calculated sizes from dimensions and types
 *
 * ### Data Integrity Criteria
 * - **File Size**: Total file size must match expected size from header information
 * - **Data Alignment**: Tensor data must be properly aligned for the data type
 * - **No Gaps**: No unexpected gaps or padding between data sections
 * - **No Overlaps**: Tensor data sections must not overlap
 * - **Accessibility**: All referenced data must be accessible within the file
 *
 * ## Error Detection Capabilities
 *
 * ### Format Errors
 * - Invalid magic number or unsupported version
 * - Malformed header structure or inconsistent counts
 * - Invalid metadata key-value pairs or unsupported data types
 * - Incorrect tensor definitions or invalid dimension specifications
 * - Improper file structure or section ordering
 *
 * ### Data Corruption
 * - Truncated files or incomplete data sections
 * - Invalid offsets pointing outside file boundaries
 * - Corrupted metadata or tensor data
 * - Inconsistent checksums or validation failures
 * - Unexpected file size or missing data
 *
 * ### Compatibility Issues
 * - Unsupported GGUF features or extensions
 * - Incompatible metadata formats or missing required fields
 * - Performance-impacting structural issues
 * - Version compatibility problems
 * - Platform-specific encoding issues
 *
 * ## Integration with Validation Framework
 *
 * This module integrates seamlessly with the broader validation framework:
 * - **Common Interface**: Uses standard validation result codes and error reporting
 * - **Error Propagation**: Detailed error messages propagated to validation reports
 * - **Performance Metrics**: Validation timing and performance statistics
 * - **Batch Processing**: Support for validating multiple GGUF files efficiently
 * - **Configuration**: Configurable validation strictness and criteria
 *
 * ## Usage Patterns
 *
 * ### Basic GGUF Validation
 * ```c
 * enum test_data_validation_result result = validate_gguf_test_file("/path/to/test.gguf");
 * if (result != TEST_DATA_VALIDATION_SUCCESS) {
 *     printf("GGUF validation failed: %s\n", test_data_validation_result_to_string(result));
 * }
 * ```
 *
 * ### Test Data Creation
 * ```c
 * // Create minimal valid GGUF test dataset
 * if (!create_minimal_gguf_dataset("/path/to/test.gguf", 100, 512)) {
 *     printf("Failed to create GGUF test dataset\n");
 * }
 * 
 * // Create corrupted file for error testing
 * if (!create_corrupted_gguf_test_file("/path/to/corrupted.gguf")) {
 *     printf("Failed to create corrupted GGUF test file\n");
 * }
 * ```
 *
 * ### Integration with Test Suite
 * ```c
 * struct test_data_validation_report report;
 * if (!validate_all_test_data("/test/data", &report)) {
 *     // GGUF-specific errors will be included in the report
 *     print_validation_report(&report);
 * }
 * ```
 *
 * ## Performance Considerations
 *
 * ### Validation Performance
 * - **Fast Header Validation**: Quick validation of header structure and metadata
 * - **Selective Data Validation**: Option to validate only headers for performance
 * - **Streaming Validation**: Memory-efficient validation for large files
 * - **Parallel Processing**: Support for concurrent validation of multiple files
 * - **Caching**: Validation result caching for repeated validations
 *
 * ### Memory Usage
 * - **Minimal Memory**: Validation requires minimal memory overhead
 * - **Streaming Access**: Large files validated without loading entirely into memory
 * - **Resource Management**: Automatic cleanup of validation resources
 * - **Memory Monitoring**: Optional memory usage tracking during validation
 *
 * ## Security Considerations
 *
 * ### Input Validation
 * - **Bounds Checking**: All file access operations include proper bounds checking
 * - **Integer Overflow**: Protection against integer overflow in size calculations
 * - **Buffer Safety**: Safe buffer handling for metadata and tensor data
 * - **Path Validation**: Secure file path handling and validation
 *
 * ### Error Handling
 * - **Graceful Degradation**: Validation failures do not crash the system
 * - **Resource Cleanup**: Proper cleanup even in error conditions
 * - **Error Isolation**: Validation errors do not affect other components
 * - **Detailed Logging**: Comprehensive error logging for security analysis
 *
 * @see test-data-validator.h for the main validation interface
 * @see test-data-validator-common.h for shared validation utilities
 * @see formats/gguf/llama-dataset-gguf.h for GGUF format implementation
 * @see validation/llama-dataset-validation.h for core validation infrastructure
 *
 * @author Dataset Converter Team
 * @version 1.0
 * @since 2024
 */

#include "test-data-validator-common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Validate a GGUF test data file with comprehensive format checking.
 *
 * Performs thorough validation of a GGUF test data file, checking all aspects of
 * the format including header structure, metadata consistency, tensor definitions,
 * and data integrity. This function implements GGUF-specific validation logic
 * tailored to the requirements of the dataset converter test suite.
 *
 * ## Validation Process
 *
 * The validation process follows a structured approach:
 *
 * ### 1. File Accessibility
 * - Verifies file exists and is readable
 * - Checks file permissions and accessibility
 * - Validates file size is non-zero and reasonable
 * - Ensures file is not locked or in use
 *
 * ### 2. Header Validation
 * - **Magic Number**: Verifies GGUF magic number (0x46554747)
 * - **Version Check**: Ensures GGUF version is supported (1, 2, or 3)
 * - **Count Validation**: Validates tensor count and metadata count are reasonable
 * - **Header Integrity**: Checks header structure and field consistency
 * - **Size Calculation**: Verifies header size matches expected size
 *
 * ### 3. Metadata Validation
 * - **Key Validation**: Ensures metadata keys are valid UTF-8 strings
 * - **Type Checking**: Verifies metadata value types match declared types
 * - **Required Fields**: Checks for presence of essential metadata fields
 * - **Value Validation**: Validates metadata values are within expected ranges
 * - **Encoding**: Verifies proper UTF-8 encoding for string values
 *
 * ### 4. Tensor Definition Validation
 * - **Name Validation**: Ensures tensor names are valid UTF-8 strings
 * - **Dimension Checking**: Validates tensor dimensions are positive and reasonable
 * - **Type Validation**: Verifies tensor data types are supported GGUF types
 * - **Offset Validation**: Checks tensor data offsets are valid and within file
 * - **Size Consistency**: Ensures calculated tensor sizes match declared sizes
 *
 * ### 5. Data Integrity Validation
 * - **File Size**: Verifies total file size matches expected size from metadata
 * - **Data Alignment**: Checks tensor data is properly aligned for data types
 * - **Offset Consistency**: Validates all offsets point to valid file locations
 * - **No Overlaps**: Ensures tensor data sections do not overlap
 * - **Completeness**: Verifies all declared data is present and accessible
 *
 * ## GGUF-Specific Validation Criteria
 *
 * ### Header Requirements
 * - Magic number must be exactly "GGUF" (0x46554747)
 * - Version must be 1, 2, or 3 (other versions not supported)
 * - Tensor count must be >= 0 and <= MAX_REASONABLE_TENSOR_COUNT
 * - Metadata count must be >= 0 and <= MAX_REASONABLE_METADATA_COUNT
 * - Header must be properly structured with correct field ordering
 *
 * ### Metadata Requirements
 * - All keys must be valid UTF-8 strings with length > 0
 * - String values must be valid UTF-8 with reasonable length limits
 * - Numeric values must be within valid ranges for their types
 * - Array values must have consistent element types and reasonable sizes
 * - Required metadata fields must be present for dataset functionality
 *
 * ### Tensor Requirements
 * - Tensor names must be valid UTF-8 strings with length > 0
 * - Dimensions must be positive integers with reasonable limits
 * - Data types must be valid GGUF tensor types (F32, F16, Q4_0, etc.)
 * - Data offsets must point to valid locations within the file
 * - Tensor data must be properly aligned for the specified data type
 *
 * ### Test Data Specific Requirements
 * - File size must be appropriate for test data (not too large or small)
 * - Must contain at least one tensor for meaningful testing
 * - Tensor data must be accessible and not corrupted
 * - Metadata must include fields required for dataset converter functionality
 * - Performance characteristics must be suitable for test execution
 *
 * ## Error Detection and Reporting
 *
 * The function detects and reports various types of errors:
 *
 * ### Format Errors
 * - `TEST_DATA_VALIDATION_INVALID_FORMAT`: Invalid GGUF magic number or version
 * - `TEST_DATA_VALIDATION_CORRUPTED`: Corrupted header, metadata, or tensor data
 * - `TEST_DATA_VALIDATION_INCOMPLETE`: Truncated file or missing data sections
 *
 * ### Content Errors
 * - `TEST_DATA_VALIDATION_INVALID_CONTENT`: Invalid metadata values or tensor definitions
 * - `TEST_DATA_VALIDATION_INCONSISTENT`: Inconsistent metadata or tensor information
 * - `TEST_DATA_VALIDATION_UNSUPPORTED`: Unsupported GGUF features or data types
 *
 * ### Access Errors
 * - `TEST_DATA_VALIDATION_FILE_NOT_FOUND`: File does not exist or is not accessible
 * - `TEST_DATA_VALIDATION_PERMISSION_DENIED`: Insufficient permissions to read file
 * - `TEST_DATA_VALIDATION_IO_ERROR`: File I/O errors during validation
 *
 * ## Performance Characteristics
 *
 * - **Validation Time**: Typically 1-10ms for small test files, 10-100ms for larger files
 * - **Memory Usage**: Minimal memory overhead, typically < 1MB regardless of file size
 * - **I/O Efficiency**: Optimized file access patterns to minimize disk I/O
 * - **Scalability**: Efficient validation for files from KB to GB in size
 *
 * ## Thread Safety
 *
 * This function is thread-safe and can be called concurrently from multiple threads
 * to validate different files. No shared state is modified during validation.
 *
 * @param path Path to the GGUF file to validate (must be non-NULL, valid file path)
 * @return Validation result code indicating success or specific failure type:
 *         - `TEST_DATA_VALIDATION_SUCCESS`: File is valid and suitable for testing
 *         - `TEST_DATA_VALIDATION_FILE_NOT_FOUND`: File does not exist
 *         - `TEST_DATA_VALIDATION_PERMISSION_DENIED`: Cannot read file
 *         - `TEST_DATA_VALIDATION_INVALID_FORMAT`: Not a valid GGUF file
 *         - `TEST_DATA_VALIDATION_CORRUPTED`: File is corrupted or incomplete
 *         - `TEST_DATA_VALIDATION_INVALID_CONTENT`: Content does not meet test requirements
 *         - `TEST_DATA_VALIDATION_IO_ERROR`: File I/O error during validation
 *
 * @note This function performs read-only operations and does not modify the file
 * @note Detailed error information can be obtained through the validation report system
 * @note For batch validation of multiple files, consider using validate_all_test_data()
 *
 * @see test_data_validation_result for complete list of possible return values
 * @see create_minimal_gguf_dataset() for creating valid test files
 * @see validate_all_test_data() for batch validation capabilities
 * @see formats/gguf/llama-dataset-gguf.h for GGUF format implementation details
 *
 * @warning The path parameter must point to a valid file path. Passing NULL or
 *          invalid paths will result in undefined behavior.
 */
enum test_data_validation_result validate_gguf_test_file(const char* path);

/**
 * @brief Create a minimal valid GGUF test dataset with specified characteristics.
 *
 * Creates a minimal but fully valid GGUF dataset file suitable for testing the
 * dataset converter functionality. The generated file contains the minimum required
 * structure and metadata to be recognized as a valid GGUF dataset while remaining
 * small and efficient for test execution.
 *
 * ## Generated File Structure
 *
 * The created GGUF file includes:
 *
 * ### Header Section
 * - **Magic Number**: Standard GGUF magic number (0x46554747)
 * - **Version**: Latest supported GGUF version for maximum compatibility
 * - **Tensor Count**: Set to 1 (single tensor containing all sequence data)
 * - **Metadata Count**: Minimum required metadata entries for dataset functionality
 *
 * ### Metadata Section
 * - **training.type**: Set to "dataset" to identify as training dataset
 * - **training.sequence_count**: Set to the specified num_sequences parameter
 * - **training.sequence_length**: Set to the specified sequence_length parameter
 * - **training.token_type**: Set to "int32" for standard token representation
 * - **training.format_version**: Set to current dataset format version
 * - **general.name**: Set to "minimal_test_dataset" for identification
 * - **general.description**: Brief description of the test dataset purpose
 * - **general.created**: Timestamp of dataset creation
 *
 * ### Tensor Information Section
 * - **Tensor Name**: "training.sequences" (standard name for sequence data)
 * - **Dimensions**: [num_sequences, sequence_length] for 2D sequence array
 * - **Data Type**: GGUF_TYPE_INT32 for token data
 * - **Offset**: Calculated offset to tensor data section
 * - **Size**: Calculated size based on dimensions and data type
 *
 * ### Tensor Data Section
 * - **Sequence Data**: Generated token sequences with realistic token values
 * - **Token Range**: Tokens in range [1, 32000] to simulate realistic vocabulary
 * - **Pattern**: Simple but varied patterns to enable meaningful testing
 * - **Alignment**: Proper data alignment for optimal performance
 *
 * ## Generated Content Characteristics
 *
 * ### Token Generation Strategy
 * - **Vocabulary Range**: Tokens generated in range [1, 32000] (excluding special tokens)
 * - **Pattern Variation**: Each sequence has different but predictable patterns
 * - **Realistic Distribution**: Token distribution approximates natural language patterns
 * - **Deterministic**: Same parameters always generate identical content for reproducibility
 * - **Test-Friendly**: Content designed to be easily validated and debugged
 *
 * ### Sequence Patterns
 * - **Sequence 0**: Ascending pattern: [1, 2, 3, ..., sequence_length]
 * - **Sequence 1**: Descending pattern: [sequence_length, sequence_length-1, ..., 1]
 * - **Sequence 2**: Alternating pattern: [1, sequence_length, 2, sequence_length-1, ...]
 * - **Sequence N**: Modular pattern based on sequence index for variety
 * - **Padding**: Sequences shorter than sequence_length are padded with token 0
 *
 * ## File Size and Performance
 *
 * ### Size Calculation
 * - **Header**: ~100 bytes (fixed overhead)
 * - **Metadata**: ~500 bytes (essential metadata only)
 * - **Tensor Info**: ~100 bytes (single tensor definition)
 * - **Tensor Data**: num_sequences × sequence_length × 4 bytes (int32 tokens)
 * - **Total**: ~700 bytes + (num_sequences × sequence_length × 4) bytes
 *
 * ### Performance Characteristics
 * - **Creation Time**: Typically 1-10ms for small datasets (< 1MB)
 * - **Memory Usage**: Minimal memory overhead during creation (< 1MB)
 * - **Loading Time**: Optimized for fast loading in test scenarios
 * - **Validation Time**: Quick validation due to minimal structure
 *
 * ## Validation and Compatibility
 *
 * The generated file is guaranteed to:
 * - **Pass Validation**: Successfully validate with validate_gguf_test_file()
 * - **Load Successfully**: Load correctly with llama_dataset_from_gguf()
 * - **Provide Access**: Allow access to all sequences via standard dataset API
 * - **Support Streaming**: Work correctly with streaming and caching features
 * - **Cross-Platform**: Work identically across different platforms and architectures
 *
 * ## Error Handling
 *
 * The function handles various error conditions:
 * - **Invalid Parameters**: Validates input parameters for reasonableness
 * - **File Creation**: Handles file creation errors and permission issues
 * - **Disk Space**: Checks for sufficient disk space before creation
 * - **Write Errors**: Handles file write errors with proper cleanup
 * - **Resource Cleanup**: Ensures proper cleanup even in error conditions
 *
 * ## Usage Examples
 *
 * ### Basic Test Dataset
 * ```c
 * // Create small dataset for basic functionality testing
 * if (!create_minimal_gguf_dataset("/tmp/test_basic.gguf", 10, 20)) {
 *     printf("Failed to create basic test dataset\n");
 * }
 * ```
 *
 * ### Performance Test Dataset
 * ```c
 * // Create larger dataset for performance testing
 * if (!create_minimal_gguf_dataset("/tmp/test_perf.gguf", 1000, 512)) {
 *     printf("Failed to create performance test dataset\n");
 * }
 * ```
 *
 * ### Edge Case Testing
 * ```c
 * // Create minimal dataset for edge case testing
 * if (!create_minimal_gguf_dataset("/tmp/test_edge.gguf", 1, 1)) {
 *     printf("Failed to create edge case test dataset\n");
 * }
 * ```
 *
 * @param path Path where to create the GGUF file (must be non-NULL, valid file path)
 * @param num_sequences Number of sequences to create (must be > 0, recommended: 1-10000)
 * @param sequence_length Length of each sequence in tokens (must be > 0, recommended: 1-2048)
 * @return true on successful creation, false on error
 *
 * @note The created file will overwrite any existing file at the specified path
 * @note Directory containing the path must exist and be writable
 * @note File size will be approximately: 700 + (num_sequences × sequence_length × 4) bytes
 * @note Generated content is deterministic - same parameters produce identical files
 *
 * @see validate_gguf_test_file() for validating the created file
 * @see create_corrupted_gguf_test_file() for creating invalid test files
 * @see llama_dataset_from_gguf() for loading the created dataset
 * @see formats/gguf/llama-dataset-gguf.h for GGUF format details
 *
 * @warning Large values for num_sequences and sequence_length can create very large files.
 *          Ensure sufficient disk space is available before calling this function.
 * @warning The path parameter must point to a valid, writable location. The function
 *          does not create intermediate directories.
 */
bool create_minimal_gguf_dataset(const char* path, uint64_t num_sequences, int32_t sequence_length);

/**
 * @brief Create a corrupted GGUF test file for error handling and validation testing.
 *
 * Creates a deliberately corrupted GGUF file to test error handling capabilities,
 * validation robustness, and recovery mechanisms in the dataset converter. The
 * corruption is carefully designed to trigger specific error conditions while
 * remaining safe for testing purposes.
 *
 * ## Corruption Types
 *
 * The function creates different types of corruption to test various error scenarios:
 *
 * ### Header Corruption
 * - **Invalid Magic Number**: Replaces GGUF magic with invalid bytes
 * - **Unsupported Version**: Sets version to unsupported value (e.g., 999)
 * - **Invalid Counts**: Sets tensor/metadata counts to impossible values
 * - **Malformed Structure**: Corrupts header field ordering or alignment
 *
 * ### Metadata Corruption
 * - **Invalid Keys**: Creates metadata keys with invalid UTF-8 sequences
 * - **Type Mismatches**: Declares one type but stores data of another type
 * - **Truncated Values**: Cuts off metadata values in the middle
 * - **Invalid Lengths**: Sets string lengths that exceed available data
 * - **Missing Required Fields**: Omits essential metadata for dataset functionality
 *
 * ### Tensor Definition Corruption
 * - **Invalid Names**: Creates tensor names with invalid UTF-8 or null bytes
 * - **Impossible Dimensions**: Sets negative or extremely large dimension values
 * - **Invalid Types**: Uses undefined or unsupported tensor data types
 * - **Bad Offsets**: Sets tensor data offsets outside file boundaries
 * - **Size Mismatches**: Declares sizes that don't match calculated sizes
 *
 * ### Data Section Corruption
 * - **Truncated Data**: Creates file that ends before all tensor data is present
 * - **Overlapping Tensors**: Creates tensor definitions with overlapping data regions
 * - **Invalid Alignment**: Places tensor data at incorrectly aligned offsets
 * - **Garbage Data**: Fills tensor data sections with random or invalid data
 * - **Missing Data**: Creates valid headers but omits actual tensor data
 *
 * ## Corruption Strategy
 *
 * The corruption is applied strategically to maximize testing value:
 *
 * ### Deterministic Corruption
 * - **Reproducible**: Same corruption pattern generated each time for consistent testing
 * - **Targeted**: Specific corruption types designed to test particular error paths
 * - **Graduated**: Multiple levels of corruption from subtle to severe
 * - **Documented**: Each corruption type is well-documented for test analysis
 *
 * ### Safety Considerations
 * - **Contained**: Corruption is limited to the test file and does not affect system
 * - **Predictable**: Corruption patterns are known and controlled
 * - **Recoverable**: No permanent damage to test environment or data
 * - **Isolated**: Corrupted files are clearly marked and isolated from valid test data
 *
 * ## Expected Validation Results
 *
 * The corrupted file should trigger specific validation failures:
 *
 * ### Format Validation Failures
 * - `TEST_DATA_VALIDATION_INVALID_FORMAT`: For header and structure corruption
 * - `TEST_DATA_VALIDATION_CORRUPTED`: For data corruption and truncation
 * - `TEST_DATA_VALIDATION_INCOMPLETE`: For missing or truncated sections
 *
 * ### Content Validation Failures
 * - `TEST_DATA_VALIDATION_INVALID_CONTENT`: For invalid metadata or tensor definitions
 * - `TEST_DATA_VALIDATION_INCONSISTENT`: For mismatched sizes or offsets
 * - `TEST_DATA_VALIDATION_UNSUPPORTED`: For unsupported features or types
 *
 * ## Testing Applications
 *
 * ### Error Handling Testing
 * - **Graceful Degradation**: Verify system handles corruption gracefully
 * - **Error Reporting**: Test quality and accuracy of error messages
 * - **Recovery Mechanisms**: Validate error recovery and cleanup procedures
 * - **Resource Management**: Ensure no resource leaks during error handling
 *
 * ### Validation Testing
 * - **Detection Accuracy**: Verify validation correctly identifies corruption
 * - **False Positives**: Ensure validation doesn't flag valid files as corrupted
 * - **Performance**: Test validation performance with corrupted files
 * - **Robustness**: Verify validation doesn't crash on corrupted input
 *
 * ### Security Testing
 * - **Buffer Overflows**: Test protection against buffer overflow attacks
 * - **Integer Overflows**: Verify protection against integer overflow exploits
 * - **Input Validation**: Test robustness of input validation routines
 * - **Memory Safety**: Ensure memory safety with malformed input
 *
 * ## File Characteristics
 *
 * ### File Structure
 * - **Base Structure**: Starts with valid GGUF structure then introduces corruption
 * - **Size**: Typically 1-10KB (small enough for fast testing)
 * - **Corruption Location**: Corruption placed at strategic locations for maximum impact
 * - **Identifiable**: File structure makes corruption type easily identifiable
 *
 * ### Corruption Markers
 * - **Header Comments**: Metadata includes information about corruption type
 * - **Filename Convention**: Suggested naming includes corruption type indicator
 * - **Documentation**: Corruption details documented for test analysis
 * - **Version Info**: Includes information about corruption generation version
 *
 * ## Usage Examples
 *
 * ### Basic Error Testing
 * ```c
 * // Create corrupted file for basic error handling tests
 * if (!create_corrupted_gguf_test_file("/tmp/test_corrupted.gguf")) {
 *     printf("Failed to create corrupted test file\n");
 * }
 * 
 * // Verify it fails validation as expected
 * enum test_data_validation_result result = validate_gguf_test_file("/tmp/test_corrupted.gguf");
 * assert(result != TEST_DATA_VALIDATION_SUCCESS);
 * ```
 *
 * ### Validation Robustness Testing
 * ```c
 * // Test validation with corrupted file
 * create_corrupted_gguf_test_file("/tmp/corrupted.gguf");
 * 
 * struct test_data_validation_report report;
 * validate_all_test_data("/tmp", &report);
 * 
 * // Verify corrupted file was detected
 * assert(report.failed_files > 0);
 * ```
 *
 * ### Error Recovery Testing
 * ```c
 * // Test dataset loading with corrupted file
 * create_corrupted_gguf_test_file("/tmp/bad.gguf");
 * 
 * struct llama_dataset* dataset = llama_dataset_from_gguf("/tmp/bad.gguf");
 * assert(dataset == NULL); // Should fail to load
 * 
 * // Verify error message is informative
 * const char* error = llama_dataset_get_error_message();
 * assert(error != NULL && strlen(error) > 0);
 * ```
 *
 * @param path Path where to create the corrupted file (must be non-NULL, valid file path)
 * @return true on successful creation of corrupted file, false on error
 *
 * @note The created file is intentionally invalid and should fail all validation checks
 * @note File will overwrite any existing file at the specified path
 * @note Directory containing the path must exist and be writable
 * @note Created file should be deleted after testing to avoid confusion
 *
 * @see validate_gguf_test_file() for validating the corrupted file (should fail)
 * @see create_minimal_gguf_dataset() for creating valid test files
 * @see test_data_validation_result for expected validation failure codes
 * @see validation/llama-dataset-validation.h for validation infrastructure
 *
 * @warning This function creates intentionally corrupted files for testing purposes only.
 *          Do not use these files for any purpose other than testing error handling.
 * @warning The corrupted file may trigger security warnings or antivirus alerts due to
 *          its intentionally malformed structure.
 * @warning Ensure the corrupted file is properly labeled and isolated to prevent
 *          accidental use as valid test data.
 */
bool create_corrupted_gguf_test_file(const char* path);

#ifdef __cplusplus
}
#endif
