#include <cstdio>
#include <cstring>
#include <string>

#include "llama-dataset-gguf.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-utils.h"
#include "llama-dataset.h"

// Helper function to get sequence data from Parquet dataset in streaming mode
static const int32_t* llama_dataset_get_parquet_sequence_streaming(const struct llama_dataset* dataset, uint64_t index) {
    if (!dataset) {
        llama_dataset_set_error("Dataset is null for Parquet streaming");
        return nullptr;
    }

    if (index >= dataset->n_seq) {
        llama_dataset_set_error("Sequence index out of bounds for Parquet streaming");
        return nullptr;
    }

    // Check if we already have cached streaming data for this sequence
    if (dataset->cached_tensors && dataset->cached_tensors[index] &&
        dataset->cached_tensors[index]->data) {
        return static_cast<const int32_t*>(dataset->cached_tensors[index]->data);
    }

    llama_dataset_set_error("Parquet streaming not fully implemented - requires format-specific data loading");
    return nullptr;
}

const int32_t* llama_dataset_sequence(const struct llama_dataset* dataset, uint64_t index) {
    if (!dataset) {
        llama_dataset_set_error("Dataset is null");
        return nullptr;
    }

    if (index >= dataset->n_seq) {
        llama_dataset_set_error("Sequence index out of bounds");
        return nullptr;
    }

    // Handle streaming mode for different dataset types
    if (dataset->streaming) {
        switch (dataset->type) {
            case DATASET_GGUF: {
                // For GGUF streaming, check if we already have cached data
                if (dataset->cached_tensors && dataset->cached_tensors[index] &&
                    dataset->cached_tensors[index]->data) {
                    // Return already cached streaming data
                    return static_cast<const int32_t*>(dataset->cached_tensors[index]->data);
                }

                // Load streaming data for the first time
                void* streaming_data = llama_dataset_gguf_get_tensor_data_streaming(dataset, index);
                if (!streaming_data) {
                    return nullptr;
                }

                // Cache the streaming data for future access and proper cleanup
                if (dataset->cached_tensors && dataset->cached_tensors[index]) {
                    dataset->cached_tensors[index]->data = streaming_data;
                } else {
                    // If no tensor cache entry exists, we have a problem - free the data to avoid leak
                    free(streaming_data);
                    llama_dataset_set_error("Tensor cache not properly initialized for streaming mode");
                    return nullptr;
                }

                return static_cast<const int32_t*>(streaming_data);
            }
            case DATASET_PARQUET:
                // For Parquet streaming, use the helper function
                return llama_dataset_get_parquet_sequence_streaming(dataset, index);
            case DATASET_TEXT:
                // Text datasets don't support streaming yet, fall through to non-streaming path
                break;
        }
    }

    // Non-streaming path: use cached tensors
    if (!dataset->cached_tensors || !dataset->cached_tensors[index]) {
        llama_dataset_set_error("Tensor cache not initialized or tensor not found");
        return nullptr;
    }

    return static_cast<const int32_t*>(dataset->cached_tensors[index]->data);
}

// Implementation of sequence_length function for all dataset types
int32_t llama_dataset_sequence_length(const struct llama_dataset* dataset, uint64_t index) {
    if (!dataset) {
        llama_dataset_set_error("Dataset is null");
        return 0;
    }

    if (index >= dataset->n_seq) {
        llama_dataset_set_error("Sequence index out of bounds");
        return 0;
    }

    // Handle streaming mode for different dataset types
    if (dataset->streaming) {
        switch (dataset->type) {
            case DATASET_GGUF: {
                // For GGUF streaming, get tensor size from GGUF context
                if (!dataset->ctx) {
                    llama_dataset_set_error("GGUF context is null");
                    return 0;
                }

                // Get tensor size and calculate length
                size_t tensor_size = gguf_get_tensor_size(dataset->ctx, index);
                return tensor_size / sizeof(int32_t);
            }
            case DATASET_PARQUET: {
                llama_dataset_set_error("Parquet streaming length not implemented");
                return 0;
            }
            case DATASET_TEXT:
                // Text datasets don't support streaming yet, fall through to non-streaming path
                break;
        }
    }

    // Non-streaming path: use cached tensors
    if (!dataset->cached_tensors || !dataset->cached_tensors[index]) {
        llama_dataset_set_error("Tensor cache not initialized or tensor not found");
        return 0;
    }

    return ggml_nelements(dataset->cached_tensors[index]);
}

// Implementation of sequence_tensor function for all dataset types
struct ggml_tensor* llama_dataset_sequence_tensor(const struct llama_dataset* dataset, uint64_t index) {
    if (!dataset) {
        llama_dataset_set_error("Dataset is null");
        return nullptr;
    }

    if (index >= dataset->n_seq) {
        llama_dataset_set_error("Sequence index out of bounds");
        return nullptr;
    }

    // For streaming mode, we don't have direct tensor access
    if (dataset->streaming) {
        llama_dataset_set_error("Direct tensor access not available in streaming mode");
        return nullptr;
    }

    // Non-streaming path: return cached tensor
    if (!dataset->cached_tensors || !dataset->cached_tensors[index]) {
        llama_dataset_set_error("Tensor cache not initialized or tensor not found");
        return nullptr;
    }

    return dataset->cached_tensors[index];
}
