#pragma once

/**
 * @file test-data-validator-gguf.h
 * @brief GGUF-specific test data validation functions.
 *
 * This header provides functions to validate and create GGUF test data files
 * for the dataset converter test suite.
 */

#include "test-data-validator-common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Validate a GGUF test data file.
 *
 * @param path Path to the GGUF file
 * @return Validation result code
 */
enum test_data_validation_result validate_gguf_test_file(const char* path);

/**
 * @brief Create a minimal valid GGUF test dataset.
 *
 * @param path Path where to create the GGUF file
 * @param num_sequences Number of sequences to create
 * @param sequence_length Length of each sequence
 * @return true on success, false on error
 */
bool create_minimal_gguf_dataset(const char* path, uint64_t num_sequences, int32_t sequence_length);

/**
 * @brief Create a corrupted GGUF test file for error handling tests.
 *
 * @param path Path where to create the corrupted file
 * @return true on success, false on error
 */
bool create_corrupted_gguf_test_file(const char* path);

#ifdef __cplusplus
}
#endif