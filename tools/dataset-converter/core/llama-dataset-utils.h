#pragma once

#include "llama-dataset.h"

/**
 * @brief Internal utility functions for dataset operations.
 *
 * This header contains internal utility functions used by the dataset implementation.
 * These functions are not part of the public API.
 */

/**
 * @brief Allocate and initialize a new dataset structure.
 *
 * @param type Dataset type
 * @param streaming Whether to use streaming mode
 * @return Pointer to the new dataset, or NULL on error
 */
struct llama_dataset * llama_dataset_alloc(enum dataset_type type, bool streaming);

/**
 * @brief Create and initialize a new dataset structure with default values.
 *
 * @return Pointer to the new dataset, or NULL on error
 */
struct llama_dataset * llama_dataset_create(void);

/**
 * @brief Cache tensor pointers for fast access.
 *
 * In non-streaming mode, this caches the actual tensor pointers from the GGML context.
 * In streaming mode, this caches tensor metadata but not the actual data.
 *
 * @param dataset Dataset to update
 * @return true on success, false on error
 */
bool llama_dataset_cache_tensors(struct llama_dataset * dataset);

/**
 * @brief Set an error message and code.
 *
 * @param code Error code
 * @param msg Error message
 */
void llama_dataset_set_error_with_code(enum dataset_error code, const char * msg);

/**
 * @brief Set an error message with default code (DATASET_ERROR_INVALID_PARAMETER).
 *
 * @param msg Error message
 */
void llama_dataset_set_error(const char * msg);

/**
 * @brief Validate and optimize tensor cache for variable sequence lengths.
 *
 * This function validates the tensor cache and optimizes it for efficient access
 * to sequences of variable lengths. It ensures all cached tensors are valid
 * and properly aligned for fast repeated access.
 *
 * @param dataset Dataset to validate and optimize
 * @return true on success, false on error
 */
bool llama_dataset_validate_and_optimize_tensor_cache(struct llama_dataset * dataset);

/**
 * @brief Validate GGUF conversion by comparing original and reloaded datasets.
 *
 * This function validates that a dataset converted to GGUF can be loaded correctly
 * and that the sequence data is preserved during conversion.
 *
 * @param original Original dataset
 * @param gguf_path Path to the converted GGUF file
 * @return true if validation passes, false on error
 */
bool llama_dataset_validate_gguf_conversion(struct llama_dataset * original, const char * gguf_path);

/**
 * @brief Compare two datasets for equality.
 *
 * This function compares two datasets to ensure they have the same sequence data.
 * It checks sequence count, sequence lengths, and token values.
 *
 * @param dataset1 First dataset
 * @param dataset2 Second dataset
 * @return true if datasets are equal, false otherwise
 */
bool llama_dataset_equal(struct llama_dataset * dataset1, struct llama_dataset * dataset2);

/**
 * @brief Create a tensor for a tokenized sequence with proper handling of variable lengths.
 *
 * This function creates a tensor for a tokenized sequence, handling variable sequence lengths
 * efficiently. It can optionally apply padding if needed for batch processing.
 *
 * @param ggml_ctx GGML context to create the tensor in
 * @param tokens Token data
 * @param n_tokens Number of tokens
 * @param tensor_name Name for the tensor
 * @param pad_to_length Optional padding length (0 = no padding)
 * @return Pointer to the created tensor, or NULL on error
 */
struct ggml_tensor * llama_dataset_create_sequence_tensor(struct ggml_context * ggml_ctx,
                                          const llama_token * tokens,
                                          int32_t n_tokens,
                                          const char * tensor_name,
                                          int32_t pad_to_length);
