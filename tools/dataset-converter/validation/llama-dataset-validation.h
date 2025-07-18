#pragma once

/**
 * @file llama-dataset-validation.h
 * @brief Data format validation and corruption detection utilities.
 *
 * This header provides validation functions for different dataset formats
 * to detect corruption and ensure data integrity.
 */

#include <stdint.h>

#include "llama-dataset.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Validation result structure.
 */
struct validation_result {
    bool is_valid;
    enum dataset_error error_code;
    char error_message[256];
    uint64_t file_size;
    uint32_t format_version;
};

/**
 * @brief Validate GGUF file format and detect corruption.
 *
 * @param path Path to the GGUF file
 * @param result Validation result structure
 * @return true if file is valid, false if corrupted or invalid
 */
bool llama_dataset_validate_gguf_file(const char* path, struct validation_result* result);

/**
 * @brief Validate Parquet file format and detect corruption.
 *
 * @param path Path to the Parquet file
 * @param result Validation result structure
 * @return true if file is valid, false if corrupted or invalid
 */
bool llama_dataset_validate_parquet_file(const char* path, struct validation_result* result);

/**
 * @brief Validate text file format and detect issues.
 *
 * @param path Path to the text file
 * @param result Validation result structure
 * @return true if file is valid, false if corrupted or invalid
 */
bool llama_dataset_validate_text_file(const char* path, struct validation_result* result);

/**
 * @brief Validate dataset file based on extension or content.
 *
 * @param path Path to the dataset file
 * @param result Validation result structure
 * @return true if file is valid, false if corrupted or invalid
 */
bool llama_dataset_validate_dataset_file(const char* path, struct validation_result* result);

/**
 * @brief Check if a file appears to be corrupted based on basic checks.
 *
 * @param path Path to the file
 * @return true if file appears corrupted, false otherwise
 */
bool llama_dataset_is_file_corrupted(const char* path);

/**
 * @brief Get a human-readable description of validation result.
 *
 * @param result Validation result structure
 * @return String description of the validation result
 */
const char* llama_dataset_validation_result_description(const struct validation_result* result);

#ifdef __cplusplus
}
#endif
