#pragma once
#ifdef LLAMA_DATASET_PARQUET_SUPPORT
#include "llama-dataset.h"

/**
 * @brief Parquet dataset loader implementation.
 *
 * This header contains functions for loading Parquet datasets.
 */

/**
 * @brief Load a dataset from a Parquet file.
 *
 * This function loads a dataset from a Parquet file, with an option to use streaming mode.
 * In streaming mode, data is loaded on demand, which can save memory for large datasets.
 *
 * @param path Path to the Parquet file
 * @param streaming Whether to use streaming mode
 * @return Pointer to the dataset, or NULL on error
 */
struct llama_dataset * llama_dataset_load_parquet_internal(const common_params * params);

/**
 * @brief Validate Parquet file schema.
 *
 * This function validates that a Parquet file has the expected schema for dataset loading.
 *
 * @param path Path to the Parquet file
 * @return true if schema is valid, false otherwise
 */
bool llama_dataet_validate_parquet_schema(const char * path);

/**
 * @brief Get metadata from Parquet file.
 *
 * This function extracts metadata from a Parquet file.
 *
 * @param path Path to the Parquet file
 * @param n_sequences Pointer to store number of sequences
 * @param max_length Pointer to store maximum sequence length
 * @return true if successful, false otherwise
 */
bool llama_dataset_get_parquet_metadata(const char * path, uint64_t * n_sequences, int32_t * max_length);
#endif
