#pragma once

#include "llama-dataset.h"

/**
 * @brief GGUF dataset loader implementation.
 * 
 * This header contains functions for loading GGUF datasets.
 */

/**
 * @brief Load a dataset from a GGUF file with streaming option.
 *
 * This function loads a dataset from a GGUF file, with an option to use streaming mode.
 * In streaming mode, tensor data is not loaded into memory until requested, which can
 * save memory for large datasets.
 *
 * @param path Path to the GGUF file
 * @param streaming Whether to use streaming mode
 * @return Pointer to the dataset, or NULL on error
 */
struct llama_dataset * llama_dataset_load_gguf(const char * path, bool streaming);

/**
 * @brief Get tensor data from a GGUF file in streaming mode.
 * 
 * This function loads tensor data from a GGUF file on demand in streaming mode.
 * It is used internally by the sequence() function.
 * 
 * @param dataset Dataset to query
 * @param index Index of the tensor
 * @return Pointer to the tensor data, or NULL on error
 */
void * gguf_get_tensor_data_streaming(const struct llama_dataset * dataset, uint64_t index);