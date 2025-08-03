#pragma once

/**
 * @file test-data-validator-text.h
 * @brief Text-specific test data validation and creation for dataset converter testing.
 *
 * This module provides comprehensive validation and creation capabilities specifically
 * designed for text format test data within the dataset converter test suite. It handles
 * text file validation, encoding verification, content integrity checks, and creation
 * of test datasets with various characteristics for thorough testing coverage.
 *
 * ## Key Responsibilities
 *
 * ### Text Format Validation
 * - Validates text file encoding (UTF-8, ASCII, and other supported encodings)
 * - Verifies text file structure and content organization
 * - Checks for proper line endings and text formatting
 * - Detects binary data contamination in text files
 * - Validates text file size and length constraints
 *
 * ### Content Integrity Verification
 * - Ensures text content is suitable for tokenization
 * - Validates character sets and encoding consistency
 * - Checks for malformed or corrupted text sequences
 * - Verifies text content meets training data requirements
 * - Detects truncated or incomplete text files
 *
 * ### Test Data Creation
 * - Creates minimal valid text datasets for basic testing
 * - Generates text files with specific characteristics (length, content, encoding)
 * - Creates corrupted text files for error handling validation
 * - Produces edge case test files (empty, very large, special characters)
 * - Generates text files with various encoding scenarios
 *
 * ### Tokenization Compatibility
 * - Validates text compatibility with llama tokenizers
 * - Checks for characters that may cause tokenization issues
 * - Verifies text length and sequence boundaries
 * - Ensures text content produces valid token sequences
 * - Tests tokenization performance characteristics
 *
 * ## Text Validation Criteria
 *
 * ### File-Level Validation
 * - **File Existence**: Verifies file exists and is accessible
 * - **File Size**: Checks file size is within reasonable bounds (not empty, not excessively large)
 * - **File Permissions**: Ensures file is readable by the test process
 * - **File Type**: Confirms file is a regular text file, not binary or special file type
 *
 * ### Encoding Validation
 * - **UTF-8 Compliance**: Validates proper UTF-8 encoding throughout the file
 * - **Character Set**: Ensures characters are within supported ranges
 * - **BOM Detection**: Handles and validates Byte Order Mark presence/absence
 * - **Line Endings**: Validates consistent line ending format (LF, CRLF, or CR)
 *
 * ### Content Validation
 * - **Text Structure**: Validates logical text organization and formatting
 * - **Length Constraints**: Ensures lines and overall content meet length requirements
 * - **Character Validity**: Checks for invalid or problematic characters
 * - **Tokenization Readiness**: Verifies text is suitable for model tokenization
 *
 * ### Performance Validation
 * - **Processing Speed**: Ensures text can be processed within reasonable time limits
 * - **Memory Usage**: Validates text doesn't cause excessive memory consumption
 * - **Streaming Compatibility**: Ensures text works correctly with streaming processing
 * - **Cache Efficiency**: Validates text access patterns work well with caching
 *
 * ## Integration with Validation Framework
 *
 * This module integrates seamlessly with the broader validation framework:
 * - **Core Validation**: Uses common validation infrastructure from test-data-validator-common.h
 * - **Main Validator**: Called by test-data-validator.h for comprehensive validation
 * - **Text Dataset**: Validates compatibility with formats/text/llama-dataset-text.h
 * - **Error Reporting**: Provides detailed error information through standard interfaces
 *
 * ## Error Detection and Reporting
 *
 * The module detects and reports various text-specific issues:
 * - **Encoding Errors**: Invalid UTF-8 sequences, unsupported characters
 * - **Format Issues**: Inconsistent line endings, binary data contamination
 * - **Content Problems**: Empty files, excessively long lines, invalid characters
 * - **Tokenization Issues**: Text that causes tokenizer failures or inefficiencies
 * - **Performance Problems**: Text that causes excessive processing time or memory usage
 *
 * ## Usage Patterns
 *
 * ### Basic Text Validation
 * ```c
 * enum test_data_validation_result result = validate_text_test_file("test_data.txt");
 * if (result != TEST_DATA_VALID) {
 *     printf("Text validation failed: %s\n", test_data_validation_result_to_string(result));
 * }
 * ```
 *
 * ### Test Data Creation
 * ```c
 * // Create minimal test dataset
 * if (!create_minimal_text_dataset("minimal_test.txt", 100)) {
 *     fprintf(stderr, "Failed to create minimal text dataset\n");
 * }
 * 
 * // Create corrupted file for error testing
 * if (!create_corrupted_text_test_file("corrupted_test.txt")) {
 *     fprintf(stderr, "Failed to create corrupted text file\n");
 * }
 * ```
 *
 * ### Comprehensive Validation
 * ```c
 * // Validate multiple aspects of text file
 * enum test_data_validation_result result = validate_text_test_file("dataset.txt");
 * switch (result) {
 *     case TEST_DATA_VALID:
 *         printf("Text file is valid and ready for use\n");
 *         break;
 *     case TEST_DATA_ENCODING_ERROR:
 *         printf("Text file has encoding issues\n");
 *         break;
 *     case TEST_DATA_FORMAT_ERROR:
 *         printf("Text file format is invalid\n");
 *         break;
 *     default:
 *         printf("Text validation failed: %s\n", test_data_validation_result_to_string(result));
 * }
 * ```
 *
 * ## Performance Considerations
 *
 * The text validation module is optimized for efficiency:
 * - **Streaming Validation**: Large files are validated in chunks to minimize memory usage
 * - **Early Termination**: Validation stops at first critical error for faster feedback
 * - **Efficient Encoding Check**: Uses optimized UTF-8 validation algorithms
 * - **Minimal I/O**: Reduces file I/O operations through intelligent buffering
 * - **Cache-Friendly**: Validation patterns are designed to work well with system caches
 *
 * ## Thread Safety
 *
 * All functions in this module are thread-safe and can be called concurrently:
 * - **Read-Only Operations**: Validation functions only read files, no shared state
 * - **Independent Creation**: File creation functions use unique temporary files
 * - **No Global State**: All state is local to function calls or passed as parameters
 * - **Atomic Operations**: File creation is atomic where possible
 *
 * @see test-data-validator.h for the main validation interface
 * @see test-data-validator-common.h for shared validation utilities
 * @see formats/text/llama-dataset-text.h for text dataset processing
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
 * @brief Validate a text test data file with comprehensive checks.
 *
 * Performs thorough validation of a text file to ensure it meets all requirements
 * for use as test data in the dataset converter test suite. This includes checking
 * file accessibility, encoding validity, content structure, and tokenization
 * compatibility.
 *
 * ## Validation Process
 *
 * The validation process consists of multiple stages:
 * 1. **File Access Validation**: Checks file existence, permissions, and basic properties
 * 2. **Encoding Validation**: Verifies UTF-8 encoding compliance and character validity
 * 3. **Content Structure Validation**: Checks text organization, line endings, and format
 * 4. **Tokenization Compatibility**: Validates text is suitable for model tokenization
 * 5. **Performance Validation**: Ensures text can be processed efficiently
 *
 * ## Specific Checks Performed
 *
 * ### File-Level Checks
 * - File exists and is accessible for reading
 * - File is a regular file (not directory, device, or special file)
 * - File size is within reasonable bounds (not empty, not excessively large)
 * - File permissions allow read access
 *
 * ### Encoding and Format Checks
 * - Valid UTF-8 encoding throughout the entire file
 * - No invalid byte sequences or encoding errors
 * - Consistent line ending format (detects mixed line endings)
 * - No binary data contamination in text content
 * - Proper handling of Byte Order Mark (BOM) if present
 *
 * ### Content Quality Checks
 * - Lines are not excessively long (configurable limit, typically 10KB per line)
 * - File contains meaningful text content (not just whitespace)
 * - No control characters that could interfere with processing
 * - Text structure is suitable for training data preparation
 *
 * ### Tokenization Compatibility Checks
 * - Text can be successfully tokenized by standard llama tokenizers
 * - No characters that cause tokenization failures or warnings
 * - Text length and structure produce reasonable token sequences
 * - Performance characteristics are acceptable for training use
 *
 * ## Return Values
 *
 * The function returns specific validation result codes:
 * - **TEST_DATA_VALID**: File passes all validation checks
 * - **TEST_DATA_FILE_NOT_FOUND**: File does not exist or is not accessible
 * - **TEST_DATA_PERMISSION_ERROR**: Insufficient permissions to read the file
 * - **TEST_DATA_ENCODING_ERROR**: Invalid UTF-8 encoding or character issues
 * - **TEST_DATA_FORMAT_ERROR**: Text format issues (line endings, structure)
 * - **TEST_DATA_CONTENT_ERROR**: Content quality issues (empty, invalid characters)
 * - **TEST_DATA_SIZE_ERROR**: File size outside acceptable bounds
 * - **TEST_DATA_TOKENIZATION_ERROR**: Text incompatible with tokenization
 * - **TEST_DATA_PERFORMANCE_ERROR**: Text causes performance issues
 *
 * ## Performance Characteristics
 *
 * The validation process is optimized for efficiency:
 * - **Streaming Validation**: Large files are processed in chunks
 * - **Early Termination**: Stops at first critical error for faster feedback
 * - **Memory Efficient**: Uses minimal memory regardless of file size
 * - **I/O Optimized**: Minimizes file I/O operations through buffering
 *
 * ## Usage Examples
 *
 * ### Basic Validation
 * ```c
 * enum test_data_validation_result result = validate_text_test_file("training_data.txt");
 * if (result == TEST_DATA_VALID) {
 *     printf("Text file is valid for testing\n");
 * } else {
 *     printf("Validation failed: %s\n", test_data_validation_result_to_string(result));
 * }
 * ```
 *
 * ### Detailed Error Handling
 * ```c
 * enum test_data_validation_result result = validate_text_test_file("dataset.txt");
 * switch (result) {
 *     case TEST_DATA_VALID:
 *         printf("File is ready for use in tests\n");
 *         break;
 *     case TEST_DATA_ENCODING_ERROR:
 *         printf("File has UTF-8 encoding issues - please fix encoding\n");
 *         break;
 *     case TEST_DATA_FORMAT_ERROR:
 *         printf("File has format issues - check line endings and structure\n");
 *         break;
 *     case TEST_DATA_SIZE_ERROR:
 *         printf("File size is outside acceptable bounds\n");
 *         break;
 *     default:
 *         printf("Validation failed: %s\n", test_data_validation_result_to_string(result));
 * }
 * ```
 *
 * @param path Path to the text file to validate. Must be a valid file path
 *             pointing to a readable text file. The path can be relative or
 *             absolute, and the file must exist and be accessible.
 *
 * @return Validation result code indicating the outcome of the validation process.
 *         Returns TEST_DATA_VALID if the file passes all checks, or a specific
 *         error code indicating the type of validation failure encountered.
 *
 * @note This function is thread-safe and can be called concurrently on different files.
 * @note The validation process does not modify the input file in any way.
 * @note For very large files, validation may take some time but uses minimal memory.
 *
 * @see test_data_validation_result_to_string() for converting result codes to strings
 * @see create_minimal_text_dataset() for creating valid test files
 * @see validate_text_test_file_detailed() for more detailed validation information
 */
enum test_data_validation_result validate_text_test_file(const char* path);

/**
 * @brief Create a minimal valid text test dataset with configurable content.
 *
 * Creates a text file containing a minimal but valid dataset suitable for testing
 * the text processing capabilities of the dataset converter. The generated content
 * is designed to be representative of real training data while being small enough
 * for efficient testing and validation.
 *
 * ## Generated Content Characteristics
 *
 * The created text file has the following properties:
 * - **Valid UTF-8 Encoding**: All content uses proper UTF-8 encoding
 * - **Consistent Line Endings**: Uses platform-appropriate line endings (LF on Unix, CRLF on Windows)
 * - **Meaningful Content**: Contains realistic text suitable for language model training
 * - **Tokenization Friendly**: Text is designed to tokenize efficiently with llama models
 * - **Varied Length**: Lines have varied lengths to test different scenarios
 * - **Character Diversity**: Includes various character types (letters, numbers, punctuation)
 *
 * ## Content Generation Strategy
 *
 * The function generates content using several strategies:
 * 1. **Template Sentences**: Uses a set of template sentences with variations
 * 2. **Random Combinations**: Combines words and phrases in meaningful ways
 * 3. **Length Variation**: Creates lines of different lengths for comprehensive testing
 * 4. **Character Coverage**: Ensures good coverage of common characters and symbols
 * 5. **Realistic Patterns**: Mimics patterns found in real training datasets
 *
 * ## File Structure
 *
 * The generated file follows this structure:
 * - Each line represents a separate training sequence
 * - Lines are separated by appropriate line endings for the platform
 * - No trailing whitespace or empty lines (unless specifically requested)
 * - File ends with a proper line ending
 * - Total file size is predictable based on num_lines parameter
 *
 * ## Performance Considerations
 *
 * The creation process is optimized for efficiency:
 * - **Buffered Writing**: Uses buffered I/O for efficient file writing
 * - **Memory Efficient**: Generates content on-demand without large memory buffers
 * - **Fast Generation**: Uses efficient algorithms for content generation
 * - **Atomic Creation**: Creates file atomically where possible to avoid partial files
 *
 * ## Error Handling
 *
 * The function handles various error conditions:
 * - **Path Issues**: Invalid paths, permission problems, disk space issues
 * - **I/O Errors**: File creation failures, write errors, disk full conditions
 * - **Parameter Validation**: Invalid num_lines values, NULL path parameters
 * - **System Limits**: Handles system-specific file size and path length limits
 *
 * ## Usage Examples
 *
 * ### Basic Dataset Creation
 * ```c
 * // Create a small test dataset with 50 lines
 * if (create_minimal_text_dataset("test_data.txt", 50)) {
 *     printf("Test dataset created successfully\n");
 * } else {
 *     fprintf(stderr, "Failed to create test dataset\n");
 * }
 * ```
 *
 * ### Large Dataset for Performance Testing
 * ```c
 * // Create a larger dataset for performance testing
 * if (create_minimal_text_dataset("large_test_data.txt", 10000)) {
 *     printf("Large test dataset created for performance testing\n");
 * } else {
 *     fprintf(stderr, "Failed to create large test dataset\n");
 * }
 * ```
 *
 * ### Validation After Creation
 * ```c
 * // Create dataset and validate it
 * const char* path = "validation_test.txt";
 * if (create_minimal_text_dataset(path, 100)) {
 *     enum test_data_validation_result result = validate_text_test_file(path);
 *     if (result == TEST_DATA_VALID) {
 *         printf("Created dataset passes validation\n");
 *     } else {
 *         printf("Created dataset failed validation: %s\n", 
 *                test_data_validation_result_to_string(result));
 *     }
 * }
 * ```
 *
 * ## Content Examples
 *
 * The generated content might include lines like:
 * ```
 * The quick brown fox jumps over the lazy dog.
 * Machine learning models require large amounts of training data.
 * Natural language processing involves understanding human language.
 * Deep learning networks can learn complex patterns from data.
 * Text preprocessing is an important step in NLP pipelines.
 * ```
 *
 * @param path Path where to create the text file. Must be a valid file path
 *             with write permissions. The directory must exist and be writable.
 *             If the file already exists, it will be overwritten.
 * @param num_lines Number of lines to create in the text file. Must be greater
 *                  than 0 and less than system-specific limits (typically 2^63-1).
 *                  Each line will contain meaningful text content suitable for
 *                  training data preparation.
 *
 * @return true on successful creation of the text file with the specified number
 *         of lines, false on any error condition including I/O errors, permission
 *         issues, or invalid parameters.
 *
 * @note This function is thread-safe when called with different path parameters.
 * @note The created file will be immediately available for reading and validation.
 * @note File creation is atomic where supported by the filesystem.
 *
 * @see validate_text_test_file() for validating the created file
 * @see create_corrupted_text_test_file() for creating intentionally invalid files
 * @see get_default_test_data_specs() for standard test data requirements
 *
 * @warning Existing files at the specified path will be overwritten without warning.
 */
bool create_minimal_text_dataset(const char* path, uint64_t num_lines);

/**
 * @brief Create a corrupted text test file for comprehensive error handling validation.
 *
 * Creates a deliberately corrupted or malformed text file to test error handling
 * capabilities, validation robustness, and recovery mechanisms in the dataset
 * converter. The corruption is designed to trigger specific error conditions
 * and validate that the system handles them gracefully.
 *
 * ## Corruption Types
 *
 * The function creates files with various types of corruption:
 *
 * ### Encoding Corruption
 * - **Invalid UTF-8 Sequences**: Inserts invalid byte sequences that violate UTF-8 encoding rules
 * - **Truncated Characters**: Creates incomplete multi-byte UTF-8 characters
 * - **Mixed Encodings**: Combines UTF-8 with other encodings in the same file
 * - **Invalid BOM**: Adds incorrect or malformed Byte Order Mark sequences
 *
 * ### Format Corruption
 * - **Mixed Line Endings**: Combines different line ending types (LF, CRLF, CR) inconsistently
 * - **Binary Data Injection**: Inserts binary data into what should be text content
 * - **Control Characters**: Includes problematic control characters that may cause issues
 * - **Null Bytes**: Inserts null bytes that can terminate string processing prematurely
 *
 * ### Content Corruption
 * - **Excessively Long Lines**: Creates lines that exceed reasonable length limits
 * - **Invalid Characters**: Includes characters that cause tokenization problems
 * - **Truncated Content**: Creates files that appear to be cut off mid-sentence
 * - **Empty Sections**: Includes large sections of whitespace or empty content
 *
 * ### Structural Corruption
 * - **File Truncation**: Creates files that end abruptly without proper termination
 * - **Size Mismatches**: Files that report incorrect size information
 * - **Permission Issues**: Files with unusual permission settings (where possible)
 * - **Symbolic Link Issues**: Creates problematic symbolic links (on supported systems)
 *
 * ## Testing Scenarios
 *
 * The corrupted files are designed to test various error handling scenarios:
 * - **Graceful Degradation**: System should handle errors without crashing
 * - **Error Detection**: Validation should correctly identify corruption types
 * - **Error Reporting**: Clear error messages should be provided to users
 * - **Recovery Mechanisms**: System should recover gracefully from errors
 * - **Resource Cleanup**: No resource leaks should occur during error handling
 *
 * ## Corruption Selection
 *
 * The function randomly selects from multiple corruption types to ensure
 * comprehensive testing coverage:
 * - Each call may produce a different type of corruption
 * - Multiple corruption types may be combined in a single file
 * - Corruption severity varies from subtle to obvious
 * - Some corruptions are designed to be detected early, others late in processing
 *
 * ## Safety Considerations
 *
 * The corrupted files are created safely:
 * - **Isolated Creation**: Files are created in isolation to prevent system corruption
 * - **Controlled Corruption**: Corruption is limited to the created file only
 * - **Reversible**: Corrupted files can be safely deleted without system impact
 * - **No System Impact**: Creation process does not affect system stability
 *
 * ## Usage Examples
 *
 * ### Basic Corruption Testing
 * ```c
 * // Create a corrupted file for error handling tests
 * if (create_corrupted_text_test_file("corrupted_test.txt")) {
 *     // Test validation on corrupted file
 *     enum test_data_validation_result result = validate_text_test_file("corrupted_test.txt");
 *     if (result != TEST_DATA_VALID) {
 *         printf("Corruption correctly detected: %s\n", 
 *                test_data_validation_result_to_string(result));
 *     } else {
 *         printf("ERROR: Corruption not detected!\n");
 *     }
 * }
 * ```
 *
 * ### Comprehensive Error Testing
 * ```c
 * // Test multiple corruption scenarios
 * for (int i = 0; i < 10; i++) {
 *     char filename[256];
 *     snprintf(filename, sizeof(filename), "corrupted_test_%d.txt", i);
 *     
 *     if (create_corrupted_text_test_file(filename)) {
 *         enum test_data_validation_result result = validate_text_test_file(filename);
 *         printf("Test %d: %s\n", i, test_data_validation_result_to_string(result));
 *         
 *         // Clean up
 *         unlink(filename);
 *     }
 * }
 * ```
 *
 * ### Dataset Loader Error Testing
 * ```c
 * // Test dataset loader error handling
 * if (create_corrupted_text_test_file("loader_error_test.txt")) {
 *     common_params params = {0};
 *     params.input_file = "loader_error_test.txt";
 *     
 *     struct llama_dataset* dataset = llama_dataset_from_txt(&params, model);
 *     if (dataset == NULL) {
 *         printf("Dataset loader correctly rejected corrupted file\n");
 *     } else {
 *         printf("ERROR: Dataset loader accepted corrupted file!\n");
 *         llama_dataset_free(dataset);
 *     }
 * }
 * ```
 *
 * ## Expected Validation Results
 *
 * Corrupted files should trigger specific validation failures:
 * - **TEST_DATA_ENCODING_ERROR**: For UTF-8 and character encoding issues
 * - **TEST_DATA_FORMAT_ERROR**: For line ending and format problems
 * - **TEST_DATA_CONTENT_ERROR**: For content quality and structure issues
 * - **TEST_DATA_SIZE_ERROR**: For files with size-related problems
 * - **TEST_DATA_TOKENIZATION_ERROR**: For tokenization compatibility issues
 *
 * @param path Path where to create the corrupted text file. Must be a valid
 *             file path with write permissions. The directory must exist and
 *             be writable. If the file already exists, it will be overwritten
 *             with corrupted content.
 *
 * @return true on successful creation of the corrupted file, false on any error
 *         condition including I/O errors, permission issues, or invalid parameters.
 *         Note that "success" means the corruption was successfully introduced,
 *         not that the file content is valid.
 *
 * @note This function is thread-safe when called with different path parameters.
 * @note The created file is intentionally invalid and should fail validation checks.
 * @note Different calls may produce different types of corruption for comprehensive testing.
 * @note The corrupted file should be deleted after testing to avoid confusion.
 *
 * @see validate_text_test_file() for testing corruption detection
 * @see create_minimal_text_dataset() for creating valid test files
 * @see test_data_validation_result for expected error codes
 *
 * @warning The created file is intentionally corrupted and should not be used for actual training.
 * @warning Existing files at the specified path will be overwritten with corrupted content.
 */
bool create_corrupted_text_test_file(const char* path);

#ifdef __cplusplus
}
#endif
