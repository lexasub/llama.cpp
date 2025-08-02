#pragma once
#include "llama-dataset.h"

// Forward declarations
struct gguf_context;
struct ggml_context;

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
void * llama_dataset_gguf_get_tensor_data_streaming(const struct llama_dataset * dataset, uint64_t index);
