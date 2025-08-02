#pragma once

/**
 * @file test-data-validator-parquet.h
 * @brief Parquet-specific test data validation functions.
 *
 * This header provides functions to validate and create Parquet test data files
 * for the dataset converter test suite.
 */

#include "test-data-validator-common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Validate a Parquet test data file.
 *
 * @param path Path to the Parquet file
 * @return Validation result code
 */
enum test_data_validation_result validate_parquet_test_file(const char* path);

/**
 * @brief Create a minimal valid Parquet test dataset.
 *
 * @param path Path where to create the Parquet file
 * @param num_sequences Number of sequences to create
 * @param sequence_length Length of each sequence
 * @return true on success, false on error
 */
bool create_minimal_parquet_dataset(const char* path, uint64_t num_sequences, int32_t sequence_length);

/**
 * @brief Create a corrupted Parquet test file for error handling tests.
 *
 * @param path Path where to create the corrupted file
 * @return true on success, false on error
 */
bool create_corrupted_parquet_test_file(const char* path);

#ifdef __cplusplus
}
#endif