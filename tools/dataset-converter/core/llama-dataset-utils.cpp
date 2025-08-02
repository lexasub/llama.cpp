#include "llama-dataset-utils.h"

#include "platform/platform-compat.h"
#include "common.h"
#include "common/log.h"
#include "llama-dataset-internal.h"
#include "llama-impl.h"
#include "llama.h"
#include "streaming-cache.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Thread-local error state
static THREAD_LOCAL struct {
    enum dataset_error code;
    char message[512];
    bool has_error;
} g_error_state = {DATASET_SUCCESS, {0}, false};

// Error handling functions
void llama_dataset_set_error_with_code(enum dataset_error code, const char* msg) {
    if (!msg) {
        msg = "Unknown error (null message)";
    }

    g_error_state.code = code;

    // Use safer string copying with proper bounds checking
    size_t msg_len = strlen(msg);
    size_t max_len = sizeof(g_error_state.message) - 1;

    if (msg_len > max_len) {
        // Truncate message if too long
        memcpy(g_error_state.message, msg, max_len);
        g_error_state.message[max_len] = '\0';
    } else {
        strcpy(g_error_state.message, msg);
    }

    g_error_state.has_error = true;

    LLAMA_LOG_ERROR("%s\n", g_error_state.message);
}

void llama_dataset_set_error(const char* msg) {
    llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, msg);
}

const char* llama_dataset_get_error(void) {
    return g_error_state.has_error ? g_error_state.message : nullptr;
}

bool llama_dataset_has_error(void) {
    return g_error_state.has_error;
}

const char* llama_dataset_get_error_message(void) {
    return g_error_state.has_error ? g_error_state.message : "";
}

enum dataset_error llama_dataset_get_error_code(void) {
    return g_error_state.code;
}

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

void llama_dataset_clear_error() {
    g_error_state.code = DATASET_SUCCESS;
    g_error_state.message[0] = '\0';
    g_error_state.has_error = false;
}

struct llama_dataset* llama_dataset_alloc(enum dataset_type type, bool streaming) {
    struct llama_dataset* dataset = static_cast<struct llama_dataset *>(malloc(sizeof(struct llama_dataset)));
    if (!dataset) {
        llama_dataset_set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate dataset structure");
        return nullptr;
    }

    // Initialize all fields to zero/null
    memset(dataset, 0, sizeof(struct llama_dataset));

    // Set type and streaming flag
    dataset->type = type;
    dataset->streaming = streaming;

    // Initialize tokenization fields
    dataset->model = nullptr;
    dataset->tokenizer_ctx = nullptr;
    dataset->owns_model = false;

    // Initialize streaming cache if in streaming mode
    if (streaming) {
        dataset->streaming_cache = new llama_dataset_streaming_cache(64 * 1024 * 1024); // 64MB default
        if (!dataset->streaming_cache) {
            llama_dataset_set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate streaming cache");
            free(dataset);
            return nullptr;
        }

        // Initialize optimization manager to nullptr (will be created on demand)
        dataset->optimization_manager = nullptr;
    }

    return dataset;
}

struct llama_dataset* llama_dataset_create(void) {
    return llama_dataset_alloc(DATASET_GGUF, false);
}

bool llama_dataset_cache_tensors(struct llama_dataset* dataset) {
    if (!dataset || !dataset->ctx) {
        llama_dataset_set_error("Invalid dataset for tensor caching");
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
        llama_dataset_set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate tensor cache");
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
                    llama_dataset_set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate streaming tensor placeholder");
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
        llama_dataset_set_error("Invalid parameters for GGUF validation");
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
        llama_dataset_set_error("Invalid parameters for tensor creation");
        return nullptr;
    }

    // Determine final length with padding
    int32_t final_length = (pad_to_length > 0 && pad_to_length > n_tokens) ? pad_to_length : n_tokens;

    // Create tensor
    struct ggml_tensor* tensor = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, final_length);
    if (!tensor) {
        llama_dataset_set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate tensor");
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
    // Currently only GGUF supports streaming
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
        llama_dataset_set_error("Invalid dataset for tensor cache optimization");
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
        llama_dataset_set_error("Dataset cannot be null");
        return false;
    }

    if (!model) {
        llama_dataset_set_error("Model cannot be null");
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
        llama_dataset_set_error("Dataset cannot be null");
        return false;
    }

    if (!dataset->model) {
        llama_dataset_set_error("Model must be set before initializing tokenization context");
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
        llama_dataset_set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED,
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
