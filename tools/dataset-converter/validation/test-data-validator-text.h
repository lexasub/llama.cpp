#pragma once

/**
 * @file test-data-validator-text.h
 * @brief Text-specific test data validation functions.
 *
 * This header provides functions to validate and create text test data files
 * for the dataset converter test suite.
 */

#include "test-data-validator-common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Validate a text test data file.
 *
 * @param path Path to the text file
 * @return Validation result code
 */
enum test_data_validation_result validate_text_test_file(const char* path);

/**
 * @brief Create a minimal valid text test dataset.
 *
 * @param path Path where to create the text file
 * @param num_lines Number of lines to create
 * @return true on success, false on error
 */
bool create_minimal_text_dataset(const char* path, uint64_t num_lines);

/**
 * @brief Create a corrupted text test file for error handling tests.
 *
 * @param path Path where to create the corrupted file
 * @return true on success, false on error
 */
bool create_corrupted_text_test_file(const char* path);

#ifdef __cplusplus
}
#endif