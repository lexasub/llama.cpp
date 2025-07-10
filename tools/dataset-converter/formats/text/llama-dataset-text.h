#pragma once

#include "llama-dataset.h"

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
struct llama_dataset * llama_dataset_load_text_internal(const char * path, struct llama_model * model, bool streaming);