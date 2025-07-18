#pragma once
#include <cstddef>
/**
 * @brief Utility functions for GGUF dataset handling.
 *
 * This file provides utility functions that are missing from the GGUF API
 * but needed for the dataset converter.
 */

/**
 * @brief Get the total size of the data section in a GGUF context.
 *
 * This function calculates the total size of all tensor data in the GGUF context.
 *
 * @param ctx The GGUF context
 * @return The total size of the data section in bytes
 */
size_t llama_dataset_gguf_get_data_size(const struct gguf_context * ctx);

/**
 * @brief Load all tensors from a GGUF context into a GGML context.
 *
 * This function loads all tensor data from a GGUF context into a GGML context.
 * The GGML context must have enough memory allocated to store all tensor data.
 *
 * @param gguf_ctx The GGUF context containing tensor data
 * @param ggml_ctx The GGML context to load tensors into
 * @return true if successful, false otherwise
 */
bool llama_dataset_gguf_load_tensors(const struct gguf_context * gguf_ctx, struct ggml_context * ggml_ctx);
