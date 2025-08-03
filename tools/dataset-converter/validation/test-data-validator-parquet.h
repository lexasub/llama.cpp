#pragma once

/**
 * @file test-data-validator-parquet.h
 * @brief Parquet-specific test data validation and creation for dataset converter tests.
 *
 * This module provides comprehensive validation and creation capabilities specifically
 * designed for Apache Parquet format test data files used in the dataset converter
 * test suite. It implements detailed Parquet format validation, schema verification,
 * and integrity checks that ensure test data files meet the requirements for reliable
 * and consistent testing of Parquet dataset loading and processing functionality.
 *
 * ## Key Responsibilities
 *
 * ### Parquet Format Validation
 * - Validates Parquet file headers and metadata structures for correctness
 * - Verifies Apache Arrow schema compatibility and data type consistency
 * - Checks column definitions, data types, and schema evolution compatibility
 * - Ensures proper Parquet file structure with valid row groups and pages
 * - Validates compression settings and encoding schemes used in test files
 *
 * ### Schema Verification and Analysis
 * - Analyzes Parquet schema for mixed content support (text and pre-tokenized data)
 * - Validates column naming conventions and data type mappings
 * - Checks for required columns and optional metadata fields
 * - Verifies schema compatibility with dataset converter expectations
 * - Ensures proper handling of nested data structures and complex types
 *
 * ### Data Integrity Checks
 * - Validates row group integrity and page-level checksums
 * - Checks data consistency across multiple row groups
 * - Verifies column statistics and metadata accuracy
 * - Ensures proper null value handling and missing data representation
 * - Validates data encoding and compression integrity
 *
 * ### Test Data Creation and Management
 * - Creates minimal but representative Parquet test datasets
 * - Generates files with appropriate schema for testing scenarios
 * - Supports creation of edge case test files (empty, single row, large datasets)
 * - Creates corrupted files for error handling and robustness testing
 * - Manages test data lifecycle and cleanup operations
 *
 * ## Parquet-Specific Validation Criteria
 *
 * ### File Structure Validation
 * - **Magic Number**: Verifies proper Parquet magic number (PAR1) at file start and end
 * - **Footer Structure**: Validates metadata footer structure and schema definitions
 * - **Row Groups**: Checks row group organization and column chunk structure
 * - **Page Headers**: Verifies page header integrity and encoding information
 * - **Compression**: Validates compression codec usage and decompression capability
 *
 * ### Schema Validation Requirements
 * - **Column Types**: Ensures supported data types (string, int32, int64, binary)
 * - **Text Columns**: Validates UTF-8 string columns for text content
 * - **Token Columns**: Checks binary or list columns for pre-tokenized data
 * - **Mixed Content**: Verifies proper handling of files with both text and token columns
 * - **Metadata**: Validates custom metadata fields and key-value pairs
 *
 * ### Content Validation Checks
 * - **Data Consistency**: Ensures data values match their declared types
 * - **Encoding Validation**: Verifies proper encoding schemes (PLAIN, DELTA, RLE)
 * - **Statistics Accuracy**: Checks column statistics against actual data values
 * - **Null Handling**: Validates proper null value representation and counts
 * - **String Encoding**: Ensures UTF-8 compliance for text data
 *
 * ### Performance and Size Constraints
 * - **File Size Limits**: Validates files are within acceptable size ranges for testing
 * - **Row Count Verification**: Ensures appropriate number of rows for test scenarios
 * - **Memory Usage**: Checks that test files don't exceed memory constraints
 * - **Loading Performance**: Validates files can be loaded within reasonable time limits
 * - **Streaming Compatibility**: Ensures files work with streaming access patterns
 *
 * ## Integration with Dataset Converter
 *
 * This module integrates closely with the dataset converter's Parquet support:
 *
 * ### Core Integration
 * - **llama-dataset-parquet.h**: Uses Parquet loading functions for validation
 * - **Schema Analysis**: Leverages schema analysis functions for validation
 * - **Tokenization**: Validates compatibility with tokenization workflows
 * - **Streaming Support**: Ensures test files work with streaming implementations
 *
 * ### Validation Workflow Integration
 * - **test-data-validator.h**: Provides Parquet-specific validation for main validator
 * - **test-data-validator-common.h**: Uses common validation infrastructure and reporting
 * - **Error Reporting**: Integrates with common error reporting and result aggregation
 * - **File Management**: Uses common file system utilities and directory operations
 *
 * ## Test Data Creation Specifications
 *
 * ### Minimal Dataset Creation
 * When creating minimal Parquet test datasets, the module generates files with:
 *
 * - **Basic Schema**: Simple schema with text and/or token columns
 * - **Representative Data**: Sample data that exercises key functionality
 * - **Proper Encoding**: Uses standard encoding schemes for compatibility
 * - **Valid Statistics**: Generates accurate column statistics and metadata
 * - **Compression**: Uses appropriate compression for test scenarios
 *
 * ### Corrupted File Creation
 * For error handling tests, creates files with specific corruption types:
 *
 * - **Header Corruption**: Invalid magic numbers or malformed headers
 * - **Schema Corruption**: Invalid schema definitions or type mismatches
 * - **Data Corruption**: Corrupted row groups or page data
 * - **Footer Corruption**: Invalid metadata footer or schema information
 * - **Truncation**: Incomplete files missing footer or data sections
 *
 * ## Usage Patterns and Examples
 *
 * ### Basic Parquet File Validation
 * ```c
 * enum test_data_validation_result result = validate_parquet_test_file("test.parquet");
 * if (result == TEST_DATA_VALID) {
 *     printf("Parquet file is valid for testing\n");
 * } else {
 *     printf("Validation failed: %s\n", test_data_validation_result_to_string(result));
 * }
 * ```
 *
 * ### Creating Test Datasets
 * ```c
 * // Create minimal dataset with 100 sequences of length 50
 * if (create_minimal_parquet_dataset("minimal_test.parquet", 100, 50)) {
 *     printf("Test dataset created successfully\n");
 * }
 * 
 * // Create corrupted file for error testing
 * if (create_corrupted_parquet_test_file("corrupted_test.parquet")) {
 *     printf("Corrupted test file created for error handling tests\n");
 * }
 * ```
 *
 * ### Integration with Main Validator
 * ```c
 * struct test_data_validation_report report;
 * validate_all_test_data("/path/to/test/data", &report);
 * // Parquet-specific validation is automatically included
 * ```
 *
 * ## Error Handling and Diagnostics
 *
 * ### Validation Error Categories
 * - **Format Errors**: Invalid Parquet file structure or magic numbers
 * - **Schema Errors**: Incompatible or malformed schema definitions
 * - **Data Errors**: Corrupted or inconsistent data content
 * - **Encoding Errors**: Invalid encoding schemes or compression issues
 * - **Compatibility Errors**: Files incompatible with dataset converter requirements
 *
 * ### Diagnostic Information
 * The validation functions provide detailed diagnostic information including:
 * - Specific location of validation failures within the file
 * - Expected vs. actual values for failed validation checks
 * - Schema analysis results and compatibility assessments
 * - Recommendations for fixing validation failures
 * - Performance metrics and file characteristics
 *
 * ## Performance Considerations
 *
 * ### Validation Efficiency
 * - **Lazy Loading**: Validates file structure before loading full content
 * - **Streaming Validation**: Uses streaming access for large file validation
 * - **Caching**: Caches validation results for repeated checks
 * - **Minimal Memory**: Validates files without loading entire content into memory
 * - **Early Termination**: Stops validation on first critical error when appropriate
 *
 * ### Test Data Optimization
 * - **Compact Files**: Creates test files optimized for fast loading and validation
 * - **Efficient Encoding**: Uses encoding schemes that balance size and performance
 * - **Appropriate Compression**: Selects compression levels suitable for testing
 * - **Minimal Overhead**: Reduces metadata overhead while maintaining validity
 * - **Streaming Friendly**: Creates files compatible with streaming access patterns
 *
 * ## Thread Safety and Concurrency
 *
 * All validation functions in this module are designed to be thread-safe:
 * - **Read-Only Operations**: Validation functions only read file data
 * - **No Shared State**: Functions don't maintain shared mutable state
 * - **Concurrent Validation**: Multiple files can be validated simultaneously
 * - **File Creation Safety**: File creation uses atomic operations where possible
 * - **Error Handling**: Thread-safe error reporting and result aggregation
 *
 * @author Dataset Converter Team
 * @version 1.0
 * @since 2024
 * @see test-data-validator.h Main validation interface and coordination
 * @see test-data-validator-common.h Common validation utilities and types
 * @see llama-dataset-parquet.h Parquet dataset loading and schema analysis
 * @see test-data-validator-core.h Core validation functionality and patterns
 */

#include "test-data-validator-common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Validate a Parquet test data file with comprehensive format checking.
 *
 * Performs thorough validation of a Parquet test data file to ensure it meets
 * all requirements for use in the dataset converter test suite. This function
 * implements comprehensive Parquet format validation including file structure,
 * schema verification, data integrity checks, and compatibility assessment.
 *
 * ## Validation Process
 *
 * The validation process includes multiple stages of checking:
 *
 * ### 1. Basic File Validation
 * - File existence and accessibility verification
 * - File size validation against reasonable limits for test data
 * - Basic file permission and ownership checks
 * - Initial file format detection and magic number verification
 *
 * ### 2. Parquet Structure Validation
 * - **Magic Number Check**: Verifies PAR1 magic number at file start and end
 * - **Footer Validation**: Checks metadata footer structure and integrity
 * - **Schema Validation**: Verifies schema definition and column specifications
 * - **Row Group Structure**: Validates row group organization and column chunks
 * - **Page Structure**: Checks page headers and data organization
 *
 * ### 3. Schema Compatibility Validation
 * - **Column Types**: Ensures supported data types for dataset converter
 * - **Text Columns**: Validates UTF-8 string columns for text content
 * - **Token Columns**: Checks binary/list columns for pre-tokenized data
 * - **Mixed Content**: Verifies proper handling of combined text/token columns
 * - **Required Fields**: Ensures presence of necessary columns for testing
 *
 * ### 4. Data Integrity Validation
 * - **Encoding Verification**: Checks data encoding schemes (PLAIN, DELTA, RLE)
 * - **Compression Validation**: Verifies compression codec compatibility
 * - **Statistics Accuracy**: Validates column statistics against actual data
 * - **Null Value Handling**: Checks proper null representation and counts
 * - **Data Consistency**: Ensures data values match declared types
 *
 * ### 5. Content Validation
 * - **Row Count Verification**: Ensures appropriate number of rows for testing
 * - **Data Quality**: Checks for reasonable data values and distributions
 * - **String Encoding**: Validates UTF-8 compliance for text columns
 * - **Token Format**: Verifies token data format and structure
 * - **Sequence Structure**: Validates data organization for sequence processing
 *
 * ## Validation Criteria
 *
 * ### File Structure Requirements
 * - Valid Parquet magic number (PAR1) at beginning and end
 * - Properly formatted metadata footer with schema information
 * - Valid row group and column chunk organization
 * - Correct page structure with valid headers and data
 * - Supported compression codecs (UNCOMPRESSED, SNAPPY, GZIP, LZ4)
 *
 * ### Schema Requirements
 * - At least one column with supported data type
 * - Column names following dataset converter conventions
 * - Proper data type mappings for text and token columns
 * - Valid schema metadata and custom key-value pairs
 * - Compatible schema evolution settings
 *
 * ### Data Quality Requirements
 * - Non-empty file with at least one row of data
 * - Consistent data types across all rows
 * - Valid UTF-8 encoding for string columns
 * - Reasonable data value ranges for test scenarios
 * - Proper null value handling and representation
 *
 * ### Performance Requirements
 * - File size appropriate for test scenarios (typically < 100MB)
 * - Row count suitable for testing (typically 1-10000 rows)
 * - Loading time within acceptable limits for test execution
 * - Memory usage compatible with test environment constraints
 * - Streaming access compatibility for large dataset testing
 *
 * ## Error Detection and Reporting
 *
 * The function provides detailed error detection for common issues:
 *
 * ### Format Errors
 * - Invalid or missing Parquet magic numbers
 * - Corrupted metadata footer or schema definitions
 * - Malformed row group or column chunk structures
 * - Invalid page headers or data encoding
 * - Unsupported compression codecs or encoding schemes
 *
 * ### Schema Errors
 * - Incompatible column data types for dataset converter
 * - Missing required columns or metadata fields
 * - Invalid column names or naming conventions
 * - Schema evolution incompatibilities
 * - Custom metadata format violations
 *
 * ### Data Errors
 * - Corrupted or inconsistent data values
 * - Invalid UTF-8 encoding in string columns
 * - Incorrect null value representation
 * - Data type mismatches between schema and content
 * - Invalid token format or structure
 *
 * @param path Path to the Parquet file to validate (relative or absolute)
 * @return Validation result code indicating the outcome:
 *         - TEST_DATA_VALID: File passes all validation checks
 *         - TEST_DATA_MISSING: File does not exist at specified path
 *         - TEST_DATA_CORRUPTED: File exists but contains corrupted data
 *         - TEST_DATA_INVALID_FORMAT: File is not a valid Parquet file
 *         - TEST_DATA_PERMISSION_ERROR: File access permission issues
 *         - TEST_DATA_SIZE_INVALID: File size outside acceptable range
 *         - TEST_DATA_CONTENT_INVALID: File content fails validation checks
 *
 * @note This function performs read-only operations and does not modify the file
 * @note Validation may take longer for large files due to comprehensive checking
 * @note Thread-safe and can be called concurrently for different files
 * @note Uses streaming access to minimize memory usage for large files
 *
 * @see create_minimal_parquet_dataset() For creating valid test files
 * @see test_data_validation_result_to_string() For converting result to string
 * @see llama_dataset_validate_parquet_schema() For schema-specific validation
 */
enum test_data_validation_result validate_parquet_test_file(const char* path);

/**
 * @brief Create a minimal valid Parquet test dataset with specified characteristics.
 *
 * Creates a minimal but fully functional Parquet test dataset file that meets
 * all requirements for use in the dataset converter test suite. The generated
 * file contains representative data with proper schema, encoding, and structure
 * suitable for testing Parquet loading, processing, and validation functionality.
 *
 * ## Generated File Characteristics
 *
 * ### Schema Structure
 * The created Parquet file includes a well-defined schema with:
 * - **Text Column**: UTF-8 string column containing sample text data
 * - **Token Column**: Binary or list column with pre-tokenized data (optional)
 * - **Sequence ID**: Integer column for sequence identification
 * - **Metadata**: Custom metadata fields for dataset converter compatibility
 * - **Statistics**: Accurate column statistics and null counts
 *
 * ### Data Content
 * - **Representative Text**: Sample text data suitable for tokenization testing
 * - **Varied Content**: Different text patterns to test various scenarios
 * - **Token Data**: Pre-tokenized sequences when token columns are included
 * - **Sequence Organization**: Data organized as discrete sequences for processing
 * - **UTF-8 Compliance**: All text data properly encoded in UTF-8
 *
 * ### File Structure
 * - **Single Row Group**: Efficient organization for test data size
 * - **Standard Encoding**: Uses PLAIN encoding for compatibility
 * - **No Compression**: Uncompressed for fast loading and debugging
 * - **Valid Footer**: Properly formatted metadata footer with schema
 * - **Magic Numbers**: Correct PAR1 magic numbers at start and end
 *
 * ## Data Generation Strategy
 *
 * ### Text Content Generation
 * - **Sample Sentences**: Generates realistic sample sentences for testing
 * - **Varied Length**: Text sequences of different lengths within reasonable bounds
 * - **Character Diversity**: Includes various characters to test tokenization
 * - **Language Patterns**: Uses patterns common in natural language processing
 * - **Special Cases**: Includes edge cases like empty strings and punctuation
 *
 * ### Token Data Generation
 * - **Realistic Tokens**: Generates token sequences that resemble real tokenization
 * - **Vocabulary Range**: Uses token IDs within typical vocabulary ranges
 * - **Sequence Structure**: Maintains proper sequence boundaries and organization
 * - **Length Variation**: Token sequences of varying lengths for testing
 * - **Special Tokens**: Includes special tokens (BOS, EOS, PAD) when appropriate
 *
 * ### Sequence Organization
 * - **Sequence Boundaries**: Clear delineation between different sequences
 * - **Consistent Length**: Each sequence contains the specified number of tokens/words
 * - **Sequence Metadata**: Additional metadata for sequence identification
 * - **Ordering**: Sequences organized in logical order for testing
 * - **Indexing**: Proper sequence indexing for random access testing
 *
 * ## File Format Specifications
 *
 * ### Parquet Format Compliance
 * - **Version Compatibility**: Uses Parquet format version 1.0 for broad compatibility
 * - **Schema Evolution**: Schema designed to support future extensions
 * - **Data Types**: Uses standard Parquet data types (BYTE_ARRAY, INT32, INT64)
 * - **Encoding Schemes**: Standard encoding schemes for maximum compatibility
 * - **Compression**: Configurable compression (default: uncompressed for testing)
 *
 * ### Dataset Converter Compatibility
 * - **Column Naming**: Follows dataset converter column naming conventions
 * - **Data Organization**: Structured for efficient dataset converter processing
 * - **Metadata Fields**: Includes custom metadata expected by dataset converter
 * - **Schema Validation**: Passes dataset converter schema validation requirements
 * - **Loading Compatibility**: Optimized for dataset converter loading patterns
 *
 * ## Performance Characteristics
 *
 * ### File Size Optimization
 * - **Minimal Overhead**: Reduces metadata overhead while maintaining validity
 * - **Efficient Encoding**: Uses encoding schemes that balance size and speed
 * - **Compact Structure**: Organizes data for minimal file size
 * - **Fast Loading**: Optimized for quick loading during test execution
 * - **Memory Efficiency**: Designed for low memory usage during processing
 *
 * ### Creation Performance
 * - **Fast Generation**: Efficient data generation algorithms
 * - **Minimal Dependencies**: Uses standard libraries for broad compatibility
 * - **Error Handling**: Robust error handling during file creation
 * - **Atomic Creation**: Creates files atomically to avoid partial files
 * - **Cleanup**: Proper cleanup on creation failures
 *
 * ## Error Handling and Validation
 *
 * ### Creation Validation
 * After creating the file, the function performs validation to ensure:
 * - File was created successfully and is accessible
 * - Generated content matches the specified parameters
 * - File structure is valid and can be read by Parquet libraries
 * - Schema is correct and compatible with dataset converter
 * - Data content is valid and properly encoded
 *
 * ### Error Recovery
 * - **Partial Creation**: Cleans up partially created files on error
 * - **Permission Issues**: Handles file permission and ownership problems
 * - **Disk Space**: Gracefully handles insufficient disk space
 * - **Path Issues**: Validates and handles invalid path specifications
 * - **Format Errors**: Detects and reports format generation issues
 *
 * @param path Path where to create the Parquet file (relative or absolute).
 *             The directory must exist and be writable. If the file already
 *             exists, it will be overwritten.
 * @param num_sequences Number of data sequences to create in the dataset.
 *                      Must be greater than 0. Typical values range from
 *                      5-1000 for test scenarios. Larger values create
 *                      larger files suitable for performance testing.
 * @param sequence_length Length of each sequence in tokens or words.
 *                        Must be greater than 0. Typical values range from
 *                        10-100 for test scenarios. Affects the amount of
 *                        data per sequence and total file size.
 * @return true if the Parquet file was created successfully and passes
 *         validation, false if creation failed due to errors such as:
 *         - Invalid path or insufficient permissions
 *         - Invalid parameter values (zero sequences or length)
 *         - Disk space or memory allocation failures
 *         - Parquet library or format generation errors
 *
 * @note The created file will be approximately (num_sequences * sequence_length * 20) bytes
 * @note File creation is atomic - either succeeds completely or fails with cleanup
 * @note The function validates the created file before returning success
 * @note Thread-safe for creating different files, but not for the same file path
 *
 * @see validate_parquet_test_file() For validating the created file
 * @see create_corrupted_parquet_test_file() For creating corrupted test files
 * @see llama_dataset_load_parquet() For loading the created dataset
 */
bool create_minimal_parquet_dataset(const char* path, uint64_t num_sequences, int32_t sequence_length);

/**
 * @brief Create a corrupted Parquet test file for error handling and robustness testing.
 *
 * Creates a deliberately corrupted Parquet file designed to test error handling,
 * validation robustness, and recovery mechanisms in the dataset converter. The
 * corruption is carefully crafted to trigger specific error conditions while
 * maintaining enough structure to be recognizable as a Parquet file attempt.
 *
 * ## Corruption Types and Strategies
 *
 * ### Header Corruption
 * - **Magic Number Corruption**: Modifies the PAR1 magic number at file start
 * - **Version Corruption**: Uses invalid or unsupported Parquet version numbers
 * - **Length Field Corruption**: Corrupts length fields in headers and metadata
 * - **Encoding Corruption**: Uses invalid encoding type identifiers
 * - **Checksum Corruption**: Modifies checksums to trigger integrity failures
 *
 * ### Schema Corruption
 * - **Invalid Data Types**: Uses unsupported or malformed data type definitions
 * - **Schema Inconsistency**: Creates mismatches between schema and actual data
 * - **Missing Required Fields**: Omits mandatory schema elements
 * - **Circular References**: Creates invalid schema references or dependencies
 * - **Metadata Corruption**: Corrupts custom metadata fields and key-value pairs
 *
 * ### Data Corruption
 * - **Row Group Corruption**: Corrupts row group headers or column chunk metadata
 * - **Page Corruption**: Modifies page headers or data content
 * - **Encoding Corruption**: Uses invalid encoding for data values
 * - **Compression Corruption**: Corrupts compressed data blocks
 * - **Statistics Corruption**: Creates inconsistent column statistics
 *
 * ### Footer Corruption
 * - **Footer Magic**: Corrupts the PAR1 magic number at file end
 * - **Metadata Length**: Provides incorrect metadata length values
 * - **Schema Corruption**: Corrupts the schema definition in the footer
 * - **File Metadata**: Corrupts file-level metadata and version information
 * - **Offset Corruption**: Provides invalid offsets to row groups or pages
 *
 * ### Structural Corruption
 * - **Truncation**: Creates incomplete files missing footer or data sections
 * - **Size Mismatches**: Creates files with incorrect size information
 * - **Alignment Issues**: Corrupts data alignment and padding
 * - **Ordering Problems**: Disrupts expected ordering of file components
 * - **Missing Components**: Omits required file structure elements
 *
 * ## Error Testing Scenarios
 *
 * ### Validation Error Testing
 * The corrupted file is designed to trigger specific validation errors:
 * - **Format Detection**: Tests ability to detect invalid Parquet format
 * - **Schema Validation**: Triggers schema validation error paths
 * - **Data Integrity**: Tests data corruption detection mechanisms
 * - **Error Recovery**: Validates graceful error handling and recovery
 * - **Error Reporting**: Tests detailed error message generation
 *
 * ### Loading Error Testing
 * - **Library Robustness**: Tests Parquet library error handling
 * - **Memory Safety**: Ensures no crashes or memory corruption on invalid files
 * - **Resource Cleanup**: Validates proper cleanup on loading failures
 * - **Error Propagation**: Tests error propagation through the loading stack
 * - **Fallback Mechanisms**: Tests fallback and recovery strategies
 *
 * ### Processing Error Testing
 * - **Streaming Errors**: Tests streaming processing with corrupted data
 * - **Partial Processing**: Validates handling of partially readable files
 * - **Cache Invalidation**: Tests cache behavior with corrupted data
 * - **Transaction Safety**: Ensures transactional safety during errors
 * - **Consistency Checks**: Validates consistency checking mechanisms
 *
 * ## Corruption Implementation
 *
 * ### Controlled Corruption
 * The corruption is implemented in a controlled manner to ensure:
 * - **Reproducible Errors**: Same corruption pattern produces same errors
 * - **Targeted Testing**: Specific corruption types for specific test scenarios
 * - **Safety**: Corruption doesn't affect other files or system stability
 * - **Detectability**: Corruption is detectable by validation mechanisms
 * - **Variety**: Multiple corruption types for comprehensive testing
 *
 * ### Corruption Patterns
 * - **Single Point**: Corrupts a single critical component
 * - **Multiple Points**: Corrupts multiple related components
 * - **Cascading**: Creates corruption that cascades through file structure
 * - **Subtle**: Creates subtle corruption that's hard to detect
 * - **Obvious**: Creates obvious corruption for basic error testing
 *
 * ## File Characteristics
 *
 * ### Basic Structure
 * The corrupted file maintains enough structure to be recognizable:
 * - **File Extension**: Uses .parquet extension for proper identification
 * - **Partial Headers**: Contains partial or corrupted headers
 * - **Recognizable Format**: Maintains some Parquet format characteristics
 * - **Error Triggers**: Contains specific patterns that trigger error conditions
 * - **Size Constraints**: Maintains reasonable file size for testing
 *
 * ### Corruption Metadata
 * - **Corruption Type**: Embedded metadata indicating corruption type
 * - **Expected Errors**: Documentation of expected error conditions
 * - **Test Scenarios**: Information about intended test scenarios
 * - **Recovery Hints**: Hints for testing recovery mechanisms
 * - **Validation Flags**: Flags for validation testing scenarios
 *
 * ## Usage in Test Scenarios
 *
 * ### Error Handling Tests
 * ```c
 * // Create corrupted file for testing
 * create_corrupted_parquet_test_file("corrupted_test.parquet");
 * 
 * // Test validation error detection
 * enum test_data_validation_result result = validate_parquet_test_file("corrupted_test.parquet");
 * assert(result == TEST_DATA_CORRUPTED);
 * 
 * // Test loading error handling
 * struct llama_dataset* dataset = llama_dataset_load_parquet(&params);
 * assert(dataset == NULL); // Should fail gracefully
 * ```
 *
 * ### Robustness Testing
 * - **Crash Prevention**: Ensures no crashes when processing corrupted files
 * - **Memory Leaks**: Validates no memory leaks during error handling
 * - **Resource Cleanup**: Tests proper cleanup of resources on errors
 * - **Error Messages**: Validates informative error message generation
 * - **Recovery Testing**: Tests system recovery after encountering corruption
 *
 * @param path Path where to create the corrupted Parquet file (relative or absolute).
 *             The directory must exist and be writable. If a file already exists
 *             at this path, it will be overwritten with the corrupted content.
 * @return true if the corrupted file was created successfully and contains the
 *         intended corruption patterns, false if creation failed due to:
 *         - Invalid path or insufficient file system permissions
 *         - Disk space or memory allocation failures
 *         - File system errors during creation
 *         - Internal errors in corruption pattern generation
 *
 * @note The created file is intentionally invalid and should fail validation
 * @note File creation is atomic - either succeeds completely or fails with cleanup
 * @note The corruption pattern is deterministic for reproducible testing
 * @note Thread-safe for creating different files, but not for the same file path
 * @warning This function creates intentionally corrupted files for testing only
 * @warning Do not use the created files for any purpose other than error testing
 *
 * @see validate_parquet_test_file() For testing validation error detection
 * @see create_minimal_parquet_dataset() For creating valid test files
 * @see llama_dataset_load_parquet() For testing loading error handling
 */
bool create_corrupted_parquet_test_file(const char* path);

/**
 * @brief Validate a tokenized Parquet file for tokenization accuracy.
 *
 * @param path Path to the tokenized Parquet file
 * @return Validation result code
 */
enum test_data_validation_result validate_tokenized_parquet_file(const char* path);

/**
 * @brief Create a test Parquet file with raw text data for tokenization testing.
 *
 * @param path Path where to create the Parquet file
 * @param texts Vector of text strings to include in the file
 * @return true on success, false on error
 */
bool create_test_parquet_with_text(const char* path, const char** texts, size_t num_texts);

/**
 * @brief Create a test Parquet file with mixed content (text and pre-tokenized).
 *
 * @param path Path where to create the Parquet file
 * @param num_sequences Number of sequences to create
 * @return true on success, false on error
 */
bool create_test_parquet_mixed_content(const char* path, size_t num_sequences);

/**
 * @brief Validate text-to-token conversion accuracy in a Parquet file.
 *
 * @param path Path to the Parquet file
 * @param model_path Path to the llama model for tokenization
 * @return Validation result code
 */
enum test_data_validation_result validate_text_to_token_conversion(const char* path, const char* model_path);

#ifdef __cplusplus
}
#endif
