#pragma once

#include "llama-dataset.h"

// Forward declarations
struct llama_model;
typedef int32_t llama_token;

/**
 * @brief Text dataset loader implementation.
 *
 * This header contains functions for loading text datasets.
 */

/**
 * @brief Load a dataset from a text file and tokenize it.
 *
 * This function loads a dataset from a text file, tokenizes it using the provided model,
 * and returns a dataset structure. In streaming mode, the text is tokenized on demand.
 *
 * @param path Path to the text file
 * @param model Model to use for tokenization
 * @param streaming Whether to use streaming mode
 * @return Pointer to the dataset, or NULL on error
 */
struct llama_dataset * llama_dataset_load_text_internal(const common_params * params, struct llama_model * model);

/**
 * @brief Tokenize a single line of text.
 *
 * @param model Model to use for tokenization
 * @param line Line of text to tokenize
 * @param line_len Length of the line
 * @param tokens Buffer to store tokens
 * @param n_tokens_max Maximum number of tokens
 * @return Number of tokens, or negative value on error
 */
int32_t llama_dataset_tokenize_line(struct llama_model * model, const char * line, int32_t line_len, llama_token * tokens, int32_t n_tokens_max);

/**
 * @brief Optimize text sequence cache for better performance.
 *
 * @param dataset Dataset to optimize
 * @return true on success, false on error
 */
bool llama_dataset_optimize_text_sequence_cache(struct llama_dataset * dataset);
