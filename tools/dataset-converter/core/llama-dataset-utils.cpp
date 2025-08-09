#include "llama-dataset-utils.h"
#include "llama-dataset-error.h"

#include "../platform/platform-compat.h"
#include "common.h"
#include "log.h"
#include "llama-dataset-internal.h"
#include "llama-impl.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Error handling functions are now implemented in llama-dataset-core.cpp
// This file contains utility functions that use the centralized error system

const char* llama_dataset_error_code_to_string(enum dataset_error code) {
    switch (code) {
        case DATASET_SUCCESS:
            return "Success";
        case DATASET_ERROR_FILE_NOT_FOUND:
            return "File not found";
        case DATASET_ERROR_INVALID_FORMAT:
            return "Invalid format";
        case DATASET_ERROR_MEMORY_ALLOCATION:
            return "Memory allocation failed";
        case DATASET_ERROR_TOKENIZATION_FAILED:
            return "Tokenization failed";
        case DATASET_ERROR_STREAMING_NOT_SUPPORTED:
            return "Streaming not supported";
        case DATASET_ERROR_INVALID_PARAMETER:
            return "Invalid parameter";
        case DATASET_ERROR_CONTEXT_CREATION_FAILED:
            return "Context creation failed";
        case DATASET_ERROR_IO_ERROR:
            return "I/O error";
        default:
            return "Unknown error";
    }
}

// llama_dataset_clear_error is now implemented in llama-dataset-core.cpp

// llama_dataset_alloc moved to llama-dataset-core.cpp to avoid streaming dependencies

// Forward declaration for internal allocation function
extern struct llama_dataset* llama_dataset_alloc_internal(enum dataset_type type, bool streaming);

struct llama_dataset* llama_dataset_create(void) {
    return llama_dataset_alloc_internal(DATASET_GGUF, false);
}

bool llama_dataset_cache_tensors(struct llama_dataset* dataset) {
    if (!dataset || !dataset->ctx) {
        llama_dataset_error_set_internal("Invalid dataset for tensor caching");
        return false;
    }

    // Get number of tensors
    uint64_t n_tensors = gguf_get_n_tensors(dataset->ctx);
    if (n_tensors == 0) {
        // No tensors to cache
        dataset->n_seq = 0;
        dataset->cached_tensors = nullptr;
        return true;
    }

    // Allocate tensor pointer array
    dataset->cached_tensors = static_cast<struct ggml_tensor **>(malloc(n_tensors * sizeof(struct ggml_tensor *)));
    if (!dataset->cached_tensors) {
        llama_dataset_error_set_with_code_internal(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate tensor cache");
        return false;
    }

    // Initialize all pointers to null
    memset(dataset->cached_tensors, 0, n_tensors * sizeof(struct ggml_tensor*));

    // Cache sequence count
    dataset->n_seq = n_tensors;

    // In non-streaming mode, we can cache the actual tensors
    if (!dataset->streaming && dataset->ggml_ctx) {
        for (uint64_t i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(dataset->ctx, i);
            if (name) {
                dataset->cached_tensors[i] = ggml_get_tensor(dataset->ggml_ctx, name);
                if (!dataset->cached_tensors[i]) {
                    LLAMA_LOG_WARN("Failed to find tensor '%s' in GGML context", name);
                }
            } else {
                LLAMA_LOG_WARN("Failed to get tensor name for index %zu", i);
            }
        }
    } else if (dataset->streaming) {
        // In streaming mode, create placeholder tensors for metadata
        for (uint64_t i = 0; i < n_tensors; i++) {
            const char* name = gguf_get_tensor_name(dataset->ctx, i);
            if (name) {
                // Create a minimal tensor structure for streaming mode
                // This will be used to store streaming data when loaded
                dataset->cached_tensors[i] = static_cast<struct ggml_tensor *>(calloc(1, sizeof(struct ggml_tensor)));
                if (!dataset->cached_tensors[i]) {
                    llama_dataset_error_set_with_code_internal(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate streaming tensor placeholder");
                    return false;
                }
                // Initialize tensor metadata but not data (data will be loaded on demand)
                dataset->cached_tensors[i]->data = nullptr;
            }
        }
    }

    return true;
}

// Compare two datasets for equality
bool llama_dataset_equal(struct llama_dataset* dataset1, struct llama_dataset* dataset2) {
    if (!dataset1 || !dataset2) {
        return false;
    }

    // Compare sequence counts
    uint64_t count1 = llama_dataset_n_sequences(dataset1);
    uint64_t count2 = llama_dataset_n_sequences(dataset2);

    if (count1 != count2) {
        LLAMA_LOG_WARN("Datasets have different sequence counts: %zu vs %zu\n", count1, count2);
        return false;
    }

    // Compare each sequence
    for (uint64_t i = 0; i < count1; i++) {
        int32_t len1 = llama_dataset_sequence_length(dataset1, i);
        int32_t len2 = llama_dataset_sequence_length(dataset2, i);

        if (len1 != len2) {
            LLAMA_LOG_WARN("Sequence %zu has different lengths: %d vs %d\n", i, len1, len2);
            return false;
        }

        const int32_t* seq1 = llama_dataset_sequence(dataset1, i);
        const int32_t* seq2 = llama_dataset_sequence(dataset2, i);

        if (!seq1 || !seq2) {
            LLAMA_LOG_WARN("Sequence %zu data is null\n", i);
            return false;
        }

        for (int32_t j = 0; j < len1; j++) {
            if (seq1[j] != seq2[j]) {
                LLAMA_LOG_WARN("Sequence %zu data differs at position %d: %d vs %d\n",
                              i, j, seq1[j], seq2[j]);
                return false;
            }
        }
    }

    return true;
}

bool llama_dataset_validate_gguf_conversion(struct llama_dataset* original, const char* gguf_path) {
    if (!original || !gguf_path) {
        llama_dataset_error_set_internal("Invalid parameters for GGUF validation");
        return false;
    }

    // Load the converted file
    common_params params;
    params.in_files.push_back(gguf_path);
    struct llama_dataset* converted = llama_dataset_from_gguf(&params);
    if (!converted) {
        // Error already set by from_gguf
        return false;
    }

    // Compare the datasets
    bool result = llama_dataset_equal(original, converted);

    // Clean up
    llama_dataset_free(converted);

    return result;
}

// Create a tensor for a tokenized sequence
struct ggml_tensor* llama_dataset_create_sequence_tensor(struct ggml_context* ggml_ctx,
                                         const llama_token* tokens,
                                         int32_t n_tokens,
                                         const char* tensor_name,
                                         int32_t pad_to_length) {
    if (!ggml_ctx || !tokens || n_tokens <= 0 || !tensor_name) {
        llama_dataset_error_set_internal("Invalid parameters for tensor creation");
        return nullptr;
    }

    // Determine final length with padding
    int32_t final_length = (pad_to_length > 0 && pad_to_length > n_tokens) ? pad_to_length : n_tokens;

    // Create tensor
    struct ggml_tensor* tensor = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, final_length);
    if (!tensor) {
        llama_dataset_error_set_with_code_internal(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate tensor");
        return nullptr;
    }

    // Set tensor name
    ggml_set_name(tensor, tensor_name);

    // Copy token data
    int32_t* tensor_data = static_cast<int32_t *>(tensor->data);
    memcpy(tensor_data, tokens, n_tokens * sizeof(int32_t));

    // Apply padding if needed
    if (final_length > n_tokens) {
        // Fill remaining space with padding token (0)
        memset(tensor_data + n_tokens, 0, (final_length - n_tokens) * sizeof(int32_t));
    }

    return tensor;
}

bool llama_dataset_supports_streaming(enum dataset_type type, const char* path) {
    (void)path; // Path parameter reserved for future format-specific streaming checks
    
    // Currently only GGUF and Parquet support streaming
    if (type == DATASET_GGUF || type == DATASET_PARQUET) {
        return true;
    }

    return false;
}

bool llama_dataset_is_streaming_enabled(const struct llama_dataset* dataset) {
    if (!dataset) {
        return false;
    }

    return dataset->streaming;
}

enum dataset_type llama_dataset_get_type(const struct llama_dataset* dataset) {
    if (!dataset) {
        return DATASET_GGUF; // Default
    }

    return dataset->type;
}

bool llama_dataset_validate_and_optimize_tensor_cache(struct llama_dataset* dataset) {
    if (!dataset || !dataset->cached_tensors) {
        llama_dataset_error_set_internal("Invalid dataset for tensor cache optimization");
        return false;
    }

    uint64_t n_seq = llama_dataset_n_sequences(dataset);
    bool all_valid = true;

    // Check each tensor
    for (uint64_t i = 0; i < n_seq; i++) {
        if (!dataset->cached_tensors[i]) {
            LLAMA_LOG_WARN("Tensor %zu is null in cache", i);
            all_valid = false;
        }
    }

    return all_valid;
}
// Tokenization management functions

bool llama_dataset_set_tokenization_model(struct llama_dataset * dataset,
                                         struct llama_model * model,
                                         bool take_ownership) {
    if (!dataset) {
        llama_dataset_error_set_internal("Dataset cannot be null");
        return false;
    }

    if (!model) {
        llama_dataset_error_set_internal("Model cannot be null");
        return false;
    }

    // Free existing model if we own it
    if (dataset->model && dataset->owns_model) {
        llama_model_free(dataset->model);
    }

    // Free existing tokenizer context
    if (dataset->tokenizer_ctx) {
        llama_free(dataset->tokenizer_ctx);
        dataset->tokenizer_ctx = nullptr;
    }

    // Set new model
    dataset->model = model;
    dataset->owns_model = take_ownership;

    return true;
}

bool llama_dataset_init_tokenization_context(struct llama_dataset * dataset) {
    if (!dataset) {
        llama_dataset_error_set_internal("Dataset cannot be null");
        return false;
    }

    if (!dataset->model) {
        llama_dataset_error_set_internal("Model must be set before initializing tokenization context");
        return false;
    }

    // Free existing context if it exists
    if (dataset->tokenizer_ctx) {
        llama_free(dataset->tokenizer_ctx);
        dataset->tokenizer_ctx = nullptr;
    }

    // Create context parameters for tokenization
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0; // We don't need context for tokenization
    ctx_params.n_batch = 1;
    ctx_params.n_ubatch = 1;
    ctx_params.n_seq_max = 1;
    ctx_params.no_perf = true;

    // Create tokenization context
    dataset->tokenizer_ctx = llama_init_from_model(dataset->model, ctx_params);
    if (!dataset->tokenizer_ctx) {
        llama_dataset_error_set_with_code_internal(DATASET_ERROR_CONTEXT_CREATION_FAILED,
                                        "Failed to create tokenization context");
        return false;
    }

    return true;
}

bool llama_dataset_has_tokenization(const struct llama_dataset * dataset) {
    if (!dataset) {
        return false;
    }

    return dataset->model != nullptr && dataset->tokenizer_ctx != nullptr;
}
