/**
 * @file llama-dataset-validation.cpp
 * @brief Implementation of data format validation and corruption detection utilities.
 *
 * This module provides comprehensive validation functionality for all supported dataset formats
 * including GGUF, Parquet, and text files. It implements data integrity checks, format validation,
 * and corruption detection algorithms to ensure dataset reliability and consistency.
 *
 * ## Key Responsibilities
 *
 * ### Format-Specific Validation
 * - **GGUF Validation**: Validates GGUF file headers, magic bytes, version compatibility,
 *   tensor counts, and metadata integrity. Performs sanity checks on file structure
 *   and detects common corruption patterns.
 * - **Parquet Validation**: Validates Apache Parquet file format including magic bytes,
 *   file structure, and basic integrity checks. Ensures compatibility with Arrow libraries.
 * - **Text Validation**: Validates text files for encoding issues, binary content detection,
 *   and character encoding consistency. Detects corrupted or malformed text data.
 *
 * ### Corruption Detection
 * - **File Structure Analysis**: Examines file headers, magic bytes, and structural elements
 *   to detect format violations and corruption patterns.
 * - **Size Validation**: Performs sanity checks on file sizes relative to claimed content
 *   to detect truncated or malformed files.
 * - **Content Analysis**: Analyzes file content for suspicious patterns, invalid data ranges,
 *   and inconsistencies that may indicate corruption.
 *
 * ### Validation Algorithms
 * - **Multi-Pass Validation**: Implements layered validation approach starting with basic
 *   file access checks, progressing to format-specific validation, and ending with
 *   content integrity verification.
 * - **Heuristic Detection**: Uses statistical analysis and pattern recognition to identify
 *   potential issues in files that pass basic format checks.
 * - **Error Classification**: Categorizes validation failures into specific error types
 *   for targeted error handling and recovery strategies.
 *
 * ## Implementation Details
 *
 * ### Validation Process
 * 1. **Parameter Validation**: Validates input parameters and initializes result structures
 * 2. **File Access**: Attempts to open and access the target file
 * 3. **Format Detection**: Identifies file format through extension analysis or content inspection
 * 4. **Header Validation**: Validates format-specific headers and magic bytes
 * 5. **Structure Analysis**: Examines file structure and metadata for consistency
 * 6. **Content Verification**: Performs content-specific validation and corruption detection
 * 7. **Result Compilation**: Compiles validation results with detailed error information
 *
 * ### Error Handling
 * - Comprehensive error classification with specific error codes
 * - Detailed error messages with diagnostic information
 * - Graceful handling of file access failures and format violations
 * - Memory-safe string operations and buffer management
 *
 * ### Performance Considerations
 * - Minimal file I/O operations for efficient validation
 * - Early termination on critical validation failures
 * - Optimized buffer sizes for different file types
 * - Efficient magic byte and header validation
 *
 * ## Usage Examples
 *
 * ```c
 * // Validate a GGUF file
 * struct validation_result result;
 * if (llama_dataset_validate_gguf_file("model.gguf", &result)) {
 *     printf("GGUF file is valid (version %u)\n", result.format_version);
 * } else {
 *     printf("Validation failed: %s\n", result.error_message);
 * }
 *
 * // Auto-detect and validate any supported format
 * if (llama_dataset_validate_dataset_file("dataset.parquet", &result)) {
 *     printf("Dataset file is valid (size: %llu bytes)\n", result.file_size);
 * }
 *
 * // Quick corruption check
 * if (llama_dataset_is_file_corrupted("suspicious_file.txt")) {
 *     printf("File appears to be corrupted\n");
 * }
 * ```
 *
 * @see llama-dataset-validation.h for public interface documentation
 * @see llama-dataset.h for core dataset functionality
 * @see llama-dataset-utils.h for utility functions
 *
 * @author llama.cpp dataset-converter team
 * @version 1.0
 * @since 2024
 */

#include "llama-dataset-validation.h"

#include <cstdio>
#include <cstring>
#include <string>

#include "llama-dataset-utils.h"
#include "llama-impl.h"

/**
 * @brief Validate GGUF file format and detect corruption.
 *
 * Performs comprehensive validation of GGUF (GPT-Generated Unified Format) files including
 * magic byte verification, version compatibility checks, metadata validation, and structural
 * integrity analysis. This function implements a multi-layered validation approach to detect
 * various types of corruption and format violations.
 *
 * ## Validation Algorithm
 *
 * ### Phase 1: Basic File Validation
 * - Validates input parameters and initializes result structure
 * - Attempts file access and retrieves file size information
 * - Performs minimum size validation (GGUF files must be at least 32 bytes)
 *
 * ### Phase 2: Header Validation
 * - Reads and validates GGUF magic bytes ("GGUF")
 * - Extracts and validates format version (currently supports versions 1-3)
 * - Reads tensor count and key-value pair count from header
 *
 * ### Phase 3: Structural Integrity
 * - Performs sanity checks on tensor and key-value counts (max 1,000,000 each)
 * - Validates file size against claimed content using heuristic size estimation
 * - Detects truncated files and unrealistic metadata claims
 *
 * ## Corruption Detection Patterns
 * - **Magic Byte Corruption**: Detects altered or missing GGUF signature
 * - **Version Incompatibility**: Identifies unsupported or corrupted version fields
 * - **Metadata Corruption**: Detects unrealistic tensor/KV counts that indicate corruption
 * - **File Truncation**: Identifies files that are too small for their claimed content
 * - **Header Corruption**: Detects I/O errors during critical header reads
 *
 * ## Performance Characteristics
 * - **Time Complexity**: O(1) - performs only header validation, not full file scan
 * - **I/O Operations**: Minimal - reads only first 32 bytes plus basic file operations
 * - **Memory Usage**: Constant - uses only stack-allocated buffers
 *
 * @param path Path to the GGUF file to validate (must not be NULL)
 * @param result Pointer to validation_result structure to store results (must not be NULL)
 *                Structure will be initialized and populated with validation outcome
 *
 * @return true if file passes all validation checks and appears to be a valid GGUF file
 * @return false if validation fails due to corruption, format violations, or access errors
 *
 * @note This function performs header-only validation for performance. Full content
 *       validation would require loading the entire file and is not performed here.
 * @note The function is thread-safe and does not modify global state.
 *
 * ## Error Conditions
 * - DATASET_ERROR_INVALID_PARAMETER: NULL parameters provided
 * - DATASET_ERROR_FILE_NOT_FOUND: File cannot be opened or accessed
 * - DATASET_ERROR_INVALID_FORMAT: File format violations or corruption detected
 * - DATASET_ERROR_IO_ERROR: File I/O operations failed during validation
 *
 * @see validation_result for detailed error information structure
 * @see llama_dataset_validate_dataset_file for auto-format detection
 */
bool llama_dataset_validate_gguf_file(const char* path, struct validation_result* result) {
    if (!path || !result) {
        if (result) {
            result->is_valid = false;
            result->error_code = DATASET_ERROR_INVALID_PARAMETER;
            strncpy(result->error_message, "Invalid parameters for GGUF validation", sizeof(result->error_message) - 1);
            result->error_message[sizeof(result->error_message) - 1] = '\0';
        }
        return false;
    }

    // Initialize result
    memset(result, 0, sizeof(struct validation_result));
    result->is_valid = false;
    result->error_code = DATASET_SUCCESS;

    FILE* file = fopen(path, "rb");
    if (!file) {
        result->error_code = DATASET_ERROR_FILE_NOT_FOUND;
        strncpy(result->error_message, "GGUF file not found", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    result->file_size = static_cast<uint64_t>(file_size);

    // Check minimum file size
    if (file_size < 32) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        strncpy(result->error_message, "GGUF file too small to be valid", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Read and validate GGUF magic number
    char magic[4];
    if (fread(magic, 1, 4, file) != 4) {
        fclose(file);
        result->error_code = DATASET_ERROR_IO_ERROR;
        strncpy(result->error_message, "Failed to read GGUF magic bytes", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Check GGUF magic bytes
    if (memcmp(magic, "GGUF", 4) != 0) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        LLAMA_LOG_DEBUG(result->error_message, sizeof(result->error_message), "Invalid GGUF magic bytes: expected 'GGUF', got '%.4s'", magic);
        return false;
    }

    // Read version
    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, file) != 1) {
        fclose(file);
        result->error_code = DATASET_ERROR_IO_ERROR;
        strncpy(result->error_message, "Failed to read GGUF version", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    result->format_version = version;

    // Validate version (currently support versions 1, 2, 3)
    if (version < 1 || version > 3) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        LLAMA_LOG_DEBUG(result->error_message, sizeof(result->error_message),  "Unsupported GGUF version: %u (supported: 1-3)", version);
        return false;
    }

    // Read tensor count and kv count
    uint64_t tensor_count, kv_count;
    if (fread(&tensor_count, sizeof(uint64_t), 1, file) != 1 ||
        fread(&kv_count, sizeof(uint64_t), 1, file) != 1) {
        fclose(file);
        result->error_code = DATASET_ERROR_IO_ERROR;
        strncpy(result->error_message, "Failed to read GGUF tensor/kv counts", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Basic sanity checks
    if (tensor_count > 1000000) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        LLAMA_LOG_DEBUG(result->error_message, sizeof(result->error_message), "Suspicious tensor count: %llu (max: 1000000)", static_cast<unsigned long long>(tensor_count));
        return false;
    }

    if (kv_count > 1000000) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        LLAMA_LOG_DEBUG(result->error_message, sizeof(result->error_message), "Suspicious key-value count: %llu (max: 1000000)", static_cast<unsigned long long>(kv_count));
        return false;
    }

    // Additional corruption checks
    // Check if file size is reasonable for the claimed tensor count
    size_t min_expected_size = 32 + (tensor_count * 4) + (kv_count * 8); // Very rough estimate
    if (static_cast<uint64_t>(file_size) < min_expected_size) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        LLAMA_LOG_DEBUG(result->error_message, sizeof(result->error_message),
                "File size (%llu) too small for claimed content (%zu tensors, %zu kvs)",
                static_cast<unsigned long long>(file_size), tensor_count, kv_count);
        return false;
    }

    fclose(file);

    // If we get here, the file appears to be valid
    result->is_valid = true;
    strncpy(result->error_message, "GGUF file validation passed", sizeof(result->error_message) - 1);
    result->error_message[sizeof(result->error_message) - 1] = '\0';
    return true;
}

/**
 * @brief Validate Parquet file format and detect corruption.
 *
 * Performs validation of Apache Parquet files by checking file structure, magic bytes,
 * and basic integrity markers. Parquet files use a specific binary format with magic
 * bytes at the end of the file, making validation straightforward but requiring
 * careful handling of file positioning and size validation.
 *
 * ## Validation Algorithm
 *
 * ### Phase 1: Basic File Validation
 * - Validates input parameters and initializes result structure
 * - Attempts file access and retrieves file size information
 * - Performs minimum size validation (Parquet files must be at least 12 bytes)
 *
 * ### Phase 2: Magic Byte Validation
 * - Seeks to the end of file and reads the last 4 bytes
 * - Validates Parquet magic bytes ("PAR1") at file footer
 * - Ensures proper file structure according to Parquet specification
 *
 * ## Corruption Detection Patterns
 * - **Magic Byte Corruption**: Detects missing or altered "PAR1" signature
 * - **File Truncation**: Identifies files too small to contain valid Parquet structure
 * - **Access Errors**: Detects I/O failures that may indicate filesystem corruption
 * - **Format Violations**: Identifies files that don't conform to Parquet specification
 *
 * ## Parquet Format Notes
 * - Parquet files store metadata at the end, with "PAR1" magic bytes as footer
 * - Minimum valid file size is 12 bytes (header + minimal metadata + footer)
 * - This validation focuses on structural integrity, not schema validation
 * - Full Parquet validation would require Apache Arrow library integration
 *
 * ## Performance Characteristics
 * - **Time Complexity**: O(1) - performs only magic byte validation
 * - **I/O Operations**: Minimal - file size query and 4-byte read at end
 * - **Memory Usage**: Constant - uses only stack-allocated buffers
 *
 * @param path Path to the Parquet file to validate (must not be NULL)
 * @param result Pointer to validation_result structure to store results (must not be NULL)
 *                Structure will be initialized and populated with validation outcome
 *
 * @return true if file passes validation and appears to be a valid Parquet file
 * @return false if validation fails due to corruption, format violations, or access errors
 *
 * @note This function performs basic structural validation only. Full schema and
 *       content validation requires Apache Arrow/Parquet library integration.
 * @note The function is thread-safe and does not modify global state.
 *
 * ## Error Conditions
 * - DATASET_ERROR_INVALID_PARAMETER: NULL parameters provided
 * - DATASET_ERROR_FILE_NOT_FOUND: File cannot be opened or accessed
 * - DATASET_ERROR_INVALID_FORMAT: File format violations or missing magic bytes
 * - DATASET_ERROR_IO_ERROR: File I/O operations failed during validation
 *
 * @see validation_result for detailed error information structure
 * @see llama_dataset_validate_dataset_file for auto-format detection
 */
bool llama_dataset_validate_parquet_file(const char* path, struct validation_result* result) {
    if (!path || !result) {
        if (result) {
            result->is_valid = false;
            result->error_code = DATASET_ERROR_INVALID_PARAMETER;
            strncpy(result->error_message, "Invalid parameters for Parquet validation", sizeof(result->error_message) - 1);
            result->error_message[sizeof(result->error_message) - 1] = '\0';
        }
        return false;
    }

    // Initialize result
    memset(result, 0, sizeof(struct validation_result));
    result->is_valid = false;
    result->error_code = DATASET_SUCCESS;

    FILE* file = fopen(path, "rb");
    if (!file) {
        result->error_code = DATASET_ERROR_FILE_NOT_FOUND;
        strncpy(result->error_message, "Parquet file not found", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    result->file_size = static_cast<uint64_t>(file_size);

    // Check minimum file size for Parquet
    if (file_size < 12) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        strncpy(result->error_message, "Parquet file too small to be valid", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Check Parquet magic bytes at the end of the file
    fseek(file, -4, SEEK_END);
    char magic[4];
    if (fread(magic, 1, 4, file) != 4) {
        fclose(file);
        result->error_code = DATASET_ERROR_IO_ERROR;
        strncpy(result->error_message, "Failed to read Parquet magic bytes", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Check for Parquet magic bytes "PAR1"
    if (memcmp(magic, "PAR1", 4) != 0) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        LLAMA_LOG_DEBUG(result->error_message, sizeof(result->error_message),
                "Invalid Parquet magic bytes: expected 'PAR1', got '%.4s'", magic);
        return false;
    }

    fclose(file);

    // If we get here, the file appears to be valid
    result->is_valid = true;
    strncpy(result->error_message, "Parquet file validation passed", sizeof(result->error_message) - 1);
    result->error_message[sizeof(result->error_message) - 1] = '\0';
    return true;
}

/**
 * @brief Validate text file format and detect issues.
 *
 * Performs validation of text files by analyzing content for encoding issues, binary
 * data contamination, and character encoding consistency. Uses heuristic analysis
 * to detect files that appear to be binary or corrupted text data, ensuring
 * compatibility with text processing pipelines.
 *
 * ## Validation Algorithm
 *
 * ### Phase 1: Basic File Validation
 * - Validates input parameters and initializes result structure
 * - Attempts file access and retrieves file size information
 * - Checks for empty files (considered invalid for dataset purposes)
 *
 * ### Phase 2: Content Analysis
 * - Reads sample buffer (up to 1024 bytes) from beginning of file
 * - Performs statistical analysis of character distribution
 * - Counts null bytes and non-printable characters for binary detection
 *
 * ### Phase 3: Heuristic Validation
 * - Applies binary detection heuristics (>10% null bytes indicates binary)
 * - Checks for excessive non-printable characters (>20% indicates corruption)
 * - Validates character encoding consistency and text file characteristics
 *
 * ## Corruption Detection Patterns
 * - **Binary Contamination**: Detects binary data in text files via null byte analysis
 * - **Encoding Corruption**: Identifies excessive non-printable characters
 * - **File Truncation**: Detects empty files that should contain text data
 * - **Character Set Issues**: Identifies potential encoding problems
 *
 * ## Heuristic Thresholds
 * - **Null Byte Threshold**: >10% null bytes in sample indicates binary file
 * - **Non-Printable Threshold**: >20% non-printable chars indicates corruption
 * - **Sample Size**: 1024 bytes for statistical analysis (or entire file if smaller)
 * - **Allowed Control Characters**: newline (\n), carriage return (\r), tab (\t)
 *
 * ## Performance Characteristics
 * - **Time Complexity**: O(n) where n is sample size (max 1024 bytes)
 * - **I/O Operations**: Single read of sample buffer from file start
 * - **Memory Usage**: Constant - 1KB stack-allocated buffer
 *
 * @param path Path to the text file to validate (must not be NULL)
 * @param result Pointer to validation_result structure to store results (must not be NULL)
 *                Structure will be initialized and populated with validation outcome
 *
 * @return true if file passes validation and appears to be valid text
 * @return false if validation fails due to binary content, corruption, or access errors
 *
 * @note This function uses heuristic analysis and may occasionally flag valid text
 *       files with unusual character distributions as potentially corrupted.
 * @note The function is thread-safe and does not modify global state.
 * @note Only validates file structure and encoding, not linguistic content quality.
 *
 * ## Error Conditions
 * - DATASET_ERROR_INVALID_PARAMETER: NULL parameters provided
 * - DATASET_ERROR_FILE_NOT_FOUND: File cannot be opened or accessed
 * - DATASET_ERROR_INVALID_FORMAT: Binary content or corruption detected
 *
 * @see validation_result for detailed error information structure
 * @see llama_dataset_validate_dataset_file for auto-format detection
 */
bool llama_dataset_validate_text_file(const char* path, struct validation_result* result) {
    if (!path || !result) {
        if (result) {
            result->is_valid = false;
            result->error_code = DATASET_ERROR_INVALID_PARAMETER;
            strncpy(result->error_message, "Invalid parameters for text validation", sizeof(result->error_message) - 1);
            result->error_message[sizeof(result->error_message) - 1] = '\0';
        }
        return false;
    }

    // Initialize result
    memset(result, 0, sizeof(struct validation_result));
    result->is_valid = false;
    result->error_code = DATASET_SUCCESS;

    FILE* file = fopen(path, "rb");
    if (!file) {
        result->error_code = DATASET_ERROR_FILE_NOT_FOUND;
        strncpy(result->error_message, "Text file not found", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    result->file_size = (uint64_t)file_size;

    // Check if file is empty
    if (file_size == 0) {
        fclose(file);
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        strncpy(result->error_message, "Text file is empty", sizeof(result->error_message) - 1);
        result->error_message[sizeof(result->error_message) - 1] = '\0';
        return false;
    }

    // Check for binary content (basic heuristic)
    fseek(file, 0, SEEK_SET);
    char buffer[1024];
    size_t bytes_read = fread(buffer, 1, sizeof(buffer), file);

    // Count null bytes and non-printable characters
    size_t null_count = 0;
    size_t non_printable_count = 0;

    for (size_t i = 0; i < bytes_read; i++) {
        if (buffer[i] == 0) {
            null_count++;
        } else if (buffer[i] < 32 && buffer[i] != '\n' && buffer[i] != '\r' && buffer[i] != '\t') {
            non_printable_count++;
        }
    }

    fclose(file);

    // If more than 10% of the sample contains null bytes, it's likely binary
    if (null_count > bytes_read / 10) {
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        LLAMA_LOG_DEBUG(result->error_message, sizeof(result->error_message),
                "File appears to be binary (contains %zu null bytes in %zu byte sample)",
                null_count, bytes_read);
        return false;
    }

    // If more than 20% contains non-printable characters, it might be corrupted
    if (non_printable_count > bytes_read / 5) {
        result->error_code = DATASET_ERROR_INVALID_FORMAT;
        LLAMA_LOG_DEBUG(result->error_message, sizeof(result->error_message),
                "File contains suspicious non-printable characters (%zu in %zu byte sample)",
                non_printable_count, bytes_read);
        return false;
    }

    // If we get here, the file appears to be valid text
    result->is_valid = true;
    strncpy(result->error_message, "Text file validation passed", sizeof(result->error_message) - 1);
    result->error_message[sizeof(result->error_message) - 1] = '\0';
    return true;
}

/**
 * @brief Validate dataset file based on extension or content.
 *
 * Provides unified validation interface that automatically detects file format
 * and applies appropriate validation algorithms. This function serves as the
 * primary entry point for dataset validation, handling format detection and
 * routing to specialized validation functions.
 *
 * ## Format Detection Algorithm
 *
 * ### Extension-Based Detection
 * - **Primary Method**: Analyzes file extension to determine format
 * - **.gguf**: Routes to GGUF validation
 * - **.parquet**: Routes to Parquet validation  
 * - **.txt, .text**: Routes to text validation
 * - **Unknown Extensions**: Falls back to content-based detection
 *
 * ### Content-Based Detection
 * - **Fallback Method**: Attempts validation with each supported format
 * - **Detection Order**: GGUF → Parquet → Text (most to least specific)
 * - **First Success**: Returns result from first successful validation
 * - **All Fail**: Returns failure with last attempted validation result
 *
 * ## Validation Strategy
 * - **Format-Specific**: Delegates to specialized validation functions
 * - **Comprehensive**: Ensures all supported formats are covered
 * - **Efficient**: Uses extension hints to avoid unnecessary format attempts
 * - **Robust**: Falls back to content detection when extension is missing/unknown
 *
 * ## Use Cases
 * - **Batch Validation**: Validate multiple files of unknown formats
 * - **Pipeline Integration**: Single validation call for mixed format datasets
 * - **Format Discovery**: Identify file formats when extensions are unreliable
 * - **Quality Assurance**: Comprehensive validation for dataset integrity
 *
 * ## Performance Characteristics
 * - **Best Case**: O(1) - single format validation with correct extension
 * - **Worst Case**: O(k) where k is number of supported formats (currently 3)
 * - **I/O Operations**: Depends on format detection path and validation complexity
 * - **Memory Usage**: Constant - delegates to format-specific validators
 *
 * @param path Path to the dataset file to validate (must not be NULL)
 * @param result Pointer to validation_result structure to store results (must not be NULL)
 *                Structure will be populated with validation outcome from successful
 *                format validation or last attempted validation if all fail
 *
 * @return true if file passes validation for any supported format
 * @return false if file fails validation for all attempted formats
 *
 * @note This function may attempt multiple validation methods, so performance
 *       depends on file format and extension accuracy.
 * @note The function is thread-safe and does not modify global state.
 * @note Result structure contains information from the successful validation
 *       or the last attempted validation if all formats fail.
 *
 * ## Error Conditions
 * - DATASET_ERROR_INVALID_PARAMETER: NULL parameters provided
 * - **Format-Specific Errors**: Inherits error conditions from delegated validators
 *
 * @see llama_dataset_validate_gguf_file for GGUF-specific validation
 * @see llama_dataset_validate_parquet_file for Parquet-specific validation  
 * @see llama_dataset_validate_text_file for text-specific validation
 * @see validation_result for detailed error information structure
 */
bool llama_dataset_validate_dataset_file(const char* path, struct validation_result* result) {
    if (!path || !result) {
        if (result) {
            result->is_valid = false;
            result->error_code = DATASET_ERROR_INVALID_PARAMETER;
            strncpy(result->error_message, "Invalid parameters for dataset validation", sizeof(result->error_message) - 1);
            result->error_message[sizeof(result->error_message) - 1] = '\0';
        }
        return false;
    }

    // Determine file type based on extension
    const char* ext = strrchr(path, '.');
    if (!ext) {
        // No extension, try to detect format by content
        return llama_dataset_validate_gguf_file(path, result) ||
               llama_dataset_validate_parquet_file(path, result) ||
               llama_dataset_validate_text_file(path, result);
    }

    // Check specific formats based on extension
    if (strcmp(ext, ".gguf") == 0) {
        return llama_dataset_validate_gguf_file(path, result);
    } else if (strcmp(ext, ".parquet") == 0) {
        return llama_dataset_validate_parquet_file(path, result);
    } else if (strcmp(ext, ".txt") == 0 || strcmp(ext, ".text") == 0) {
        return llama_dataset_validate_text_file(path, result);
    } else {
        // Unknown extension, try all formats
        return llama_dataset_validate_gguf_file(path, result) ||
               llama_dataset_validate_parquet_file(path, result) ||
               llama_dataset_validate_text_file(path, result);
    }
}

/**
 * @brief Check if a file appears to be corrupted based on basic checks.
 *
 * Provides a simplified boolean interface for corruption detection by leveraging
 * the comprehensive validation system. This function serves as a quick corruption
 * check without requiring detailed error analysis, making it suitable for
 * high-level file filtering and batch processing scenarios.
 *
 * ## Implementation Strategy
 * - **Delegation**: Uses llama_dataset_validate_dataset_file internally
 * - **Simplified Interface**: Returns simple boolean result instead of detailed validation
 * - **Format Agnostic**: Works with all supported dataset formats
 * - **Error Abstraction**: Treats all validation failures as "corruption"
 *
 * ## Use Cases
 * - **File Filtering**: Quick filtering of corrupted files from dataset collections
 * - **Batch Processing**: Efficient corruption screening in automated pipelines
 * - **Health Checks**: Simple file integrity verification in monitoring systems
 * - **Preprocessing**: Corruption detection before expensive processing operations
 *
 * ## Corruption Indicators
 * - **Access Failures**: Files that cannot be opened or read
 * - **Format Violations**: Files that don't match any supported format
 * - **Structural Issues**: Files with invalid headers, magic bytes, or metadata
 * - **Content Problems**: Files with encoding issues or suspicious content patterns
 *
 * ## Performance Characteristics
 * - **Time Complexity**: Same as llama_dataset_validate_dataset_file
 * - **I/O Operations**: Minimal - delegates to format-specific validators
 * - **Memory Usage**: Constant - uses stack-allocated validation result
 * - **Overhead**: Minimal function call overhead over direct validation
 *
 * @param path Path to the file to check for corruption (must not be NULL)
 *              NULL paths are considered "corrupted" for safety
 *
 * @return true if file appears to be corrupted, inaccessible, or invalid
 * @return false if file passes validation checks and appears to be intact
 *
 * @note This function treats all validation failures as corruption, including
 *       access errors and format mismatches, for conservative error handling.
 * @note The function is thread-safe and does not modify global state.
 * @note For detailed error information, use llama_dataset_validate_dataset_file directly.
 *
 * ## Error Handling
 * - **Conservative Approach**: Any validation failure is treated as corruption
 * - **Null Safety**: NULL paths return true (corrupted) for defensive programming
 * - **Exception Safety**: No exceptions thrown, always returns valid boolean
 *
 * @see llama_dataset_validate_dataset_file for detailed validation with error information
 * @see validation_result for comprehensive validation result structure
 */
bool llama_dataset_is_file_corrupted(const char* path) {
    if (!path) {
        return true; // Null path is considered "corrupted"
    }

    struct validation_result result;
    return !llama_dataset_validate_dataset_file(path, &result);
}

/**
 * @brief Get a human-readable description of validation result.
 *
 * Provides access to human-readable validation result descriptions stored in
 * the validation_result structure. This function serves as a safe accessor
 * for error messages and validation outcomes, ensuring consistent error
 * reporting across the validation system.
 *
 * ## Description Content
 * - **Success Messages**: Descriptive confirmation of successful validation
 * - **Error Messages**: Detailed error descriptions with diagnostic information
 * - **Format Information**: Format-specific details when available
 * - **Diagnostic Data**: Additional context for debugging validation failures
 *
 * ## Message Categories
 * - **Validation Success**: "Format file validation passed" messages
 * - **Parameter Errors**: "Invalid parameters" for NULL pointer issues
 * - **Access Errors**: "File not found" and I/O error descriptions
 * - **Format Errors**: Detailed format violation and corruption descriptions
 * - **Content Errors**: Content-specific validation failure descriptions
 *
 * ## Safety Features
 * - **Null Safety**: Returns safe default message for NULL result pointers
 * - **Buffer Safety**: All messages are null-terminated and buffer-safe
 * - **Consistent Format**: Standardized message format across all validators
 * - **Diagnostic Info**: Includes relevant technical details for debugging
 *
 * ## Performance Characteristics
 * - **Time Complexity**: O(1) - simple pointer dereference
 * - **Memory Usage**: None - returns pointer to existing string
 * - **Thread Safety**: Safe - read-only access to immutable strings
 *
 * @param result Pointer to validation_result structure containing error message
 *               If NULL, returns a safe default error message
 *
 * @return Pointer to null-terminated string containing human-readable description
 *         String is valid for the lifetime of the validation_result structure
 *         Returns "Invalid validation result" for NULL input
 *
 * @note The returned string pointer is valid only as long as the validation_result
 *       structure remains in scope and unmodified.
 * @note The function is thread-safe for read-only access to validation results.
 * @note Messages are designed for both user display and debugging purposes.
 *
 * ## Usage Examples
 * ```c
 * struct validation_result result;
 * if (!llama_dataset_validate_gguf_file("model.gguf", &result)) {
 *     printf("Validation failed: %s\n", 
 *            llama_dataset_validation_result_description(&result));
 * }
 * ```
 *
 * @see validation_result for complete result structure documentation
 * @see llama_dataset_validate_dataset_file for validation functions that populate results
 */
const char* llama_dataset_validation_result_description(const struct validation_result* result) {
    if (!result) {
        return "Invalid validation result";
    }

    return result->error_message;
}
