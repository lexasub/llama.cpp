#pragma once

/**
 * @file test-data-validator-common.h
 * @brief Common definitions and helper functions for test data validation.
 *
 * This header provides shared types, constants, and utility functions
 * used across all format-specific validation modules.
 */

#include <stdint.h>
#include "llama-dataset.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Test data validation result codes.
 */
enum test_data_validation_result {
    TEST_DATA_VALID = 0,                    // File is valid
    TEST_DATA_MISSING,                      // File does not exist
    TEST_DATA_CORRUPTED,                    // File exists but is corrupted
    TEST_DATA_INVALID_FORMAT,               // File format is invalid
    TEST_DATA_PERMISSION_ERROR,             // File permission issues
    TEST_DATA_SIZE_INVALID,                 // File size is invalid (too small/large)
    TEST_DATA_CONTENT_INVALID               // File content is invalid
};

/**
 * @brief Test data file information structure.
 */
struct test_data_file_info {
    const char* path;                       // File path
    enum dataset_type expected_type;        // Expected dataset type
    uint64_t min_size_bytes;               // Minimum expected file size
    uint64_t max_size_bytes;               // Maximum expected file size
    uint64_t expected_sequences;           // Expected number of sequences (0 = any)
    bool required;                         // Whether this file is required
    bool create_if_missing;                // Whether to create if missing
};

/**
 * @brief Test data validation report structure.
 */
struct test_data_validation_report {
    int total_files_checked;
    int valid_files;
    int missing_files;
    int corrupted_files;
    int created_files;
    int permission_fixes;
    char error_messages[4096];             // Concatenated error messages
};

//
// Common helper functions
//

/**
 * @brief Check if a file exists.
 *
 * @param path Path to the file
 * @return true if file exists, false otherwise
 */
bool file_exists(const char* path);

/**
 * @brief Check if a directory exists.
 *
 * @param path Path to the directory
 * @return true if directory exists, false otherwise
 */
bool directory_exists(const char* path);

/**
 * @brief Create a directory if it doesn't exist.
 *
 * @param path Path to the directory
 * @return true on success, false on error
 */
bool create_directory_if_missing(const char* path);

/**
 * @brief Get the size of a file.
 *
 * @param path Path to the file
 * @return File size in bytes, 0 if file doesn't exist or error
 */
uint64_t get_file_size(const char* path);

// Note: Permission checking and validation result string functions
// have been moved to test-data-validator-core.h

#ifdef __cplusplus
}
#endif