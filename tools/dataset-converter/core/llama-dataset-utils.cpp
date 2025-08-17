#include "llama-dataset-utils.h"
#include "llama-dataset-error.h"

#include "../platform/platform-compat.h"
#include "common.h"
#include "log.h"
#include "llama-dataset-internal.h"
// #include "llama-impl.h"  // Not needed for basic functionality
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Error handling functions are now implemented in llama-dataset-core.cpp
// This file contains utility functions that use the centralized error system

// llama_dataset_error_code_to_string is now implemented in llama-dataset-error.cpp

// llama_dataset_clear_error is now implemented in llama-dataset-core.cpp

// Internal allocation helper is now implemented in llama-dataset-core.cpp
extern "C" struct llama_dataset* llama_dataset_alloc_internal(enum dataset_type type, bool streaming);

struct llama_dataset* llama_dataset_create(void) {
    return llama_dataset_alloc_internal(DATASET_GGUF, false);
}

// Function moved to llama-dataset.cpp to avoid duplicate definitions

// Compare two datasets for equality
bool llama_dataset_equal(struct llama_dataset* dataset1, struct llama_dataset* dataset2) {
    if (!dataset1 || !dataset2) {
        return false;
    }

    // Compare sequence counts
    uint64_t count1 = llama_dataset_n_sequences(dataset1);
    uint64_t count2 = llama_dataset_n_sequences(dataset2);

    if (count1 != count2) {
        fprintf(stderr, "Warning: Datasets have different sequence counts: %zu vs %zu\n", count1, count2);
        return false;
    }

    // Compare each sequence
    for (uint64_t i = 0; i < count1; i++) {
        int32_t len1 = llama_dataset_sequence_length(dataset1, i);
        int32_t len2 = llama_dataset_sequence_length(dataset2, i);

        if (len1 != len2) {
            fprintf(stderr, "Warning: Sequence %zu has different lengths: %d vs %d\n", i, len1, len2);
            return false;
        }

        const int32_t* seq1 = llama_dataset_sequence(dataset1, i);
        const int32_t* seq2 = llama_dataset_sequence(dataset2, i);

        if (!seq1 || !seq2) {
            fprintf(stderr, "Warning: Sequence %zu data is null\n", i);
            return false;
        }

        for (int32_t j = 0; j < len1; j++) {
            if (seq1[j] != seq2[j]) {
                fprintf(stderr, "Warning: Sequence %zu data differs at position %d: %d vs %d\n",
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
    (void)type; // Type parameter reserved for future format-specific checks
    
    // For now, assume all formats support streaming through the registry system
    // This will be properly implemented when the registry interface is stabilized
    return true;
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

// Function moved to llama-dataset.cpp to avoid duplicate definitions
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
