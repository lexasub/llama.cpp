#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Format-specific include removed - using registry-based loading (Task G2 completed)
#include "llama-dataset-internal.h"
#include "llama-dataset-utils.h"
#include "llama-dataset.h"
#include "../streaming/streaming-cache.h"

// Format-specific functions now accessed through registry system (Task G2.1 completed)
// Direct external declarations removed - using registry-based loading

// Parquet support integrated through registry system

// Helper function to perform on-demand tokenization for streaming mode
static const int32_t* llama_dataset_tokenize_text_streaming(const struct llama_dataset* dataset, uint64_t index, const std::string& text) {
    if (!dataset || !dataset->model) {
        llama_dataset_set_error("Dataset or model is null for tokenization");
        return nullptr;
    }

    // Check if we already have this text tokenized in the streaming cache
    if (dataset->streaming_cache) {
        auto* cache = static_cast<llama_dataset_streaming_cache*>(dataset->streaming_cache);
        if (void * cached_tokens = cache->get_tokenized(index)) {
            return static_cast<const int32_t*>(cached_tokens);
        }
    }

    // Tokenize the text using the llama model
    std::vector<llama_token> tokens;

    // Create a temporary context if we don't have one
    struct llama_context* ctx = dataset->tokenizer_ctx;
    bool owns_temp_ctx = false;

    if (!ctx) {
        // Create temporary context for tokenization
        struct llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 512; // Small context for tokenization only
        ctx_params.n_batch = 1;
        ctx_params.n_threads = 1;
        ctx_params.embeddings = false;

        ctx = llama_init_from_model(dataset->model, ctx_params);
        if (!ctx) {
            llama_dataset_set_error("Failed to create tokenization context");
            return nullptr;
        }
        owns_temp_ctx = true;
    }

    // Tokenize the text
    tokens.resize(text.length() + 16); // Reserve space for tokens
    int n_tokens = llama_tokenize(llama_model_get_vocab(dataset->model), text.c_str(), text.length(),
                                 tokens.data(), tokens.size(), false, false);

    if (n_tokens < 0) {
        if (owns_temp_ctx) {
            llama_free(ctx);
        }
        llama_dataset_set_error("Tokenization failed - text too long or invalid");
        return nullptr;
    }

    tokens.resize(n_tokens);

    // Clean up temporary context
    if (owns_temp_ctx) {
        llama_free(ctx);
    }

    // Convert to int32_t vector for caching
    std::vector<int32_t> int32_tokens(tokens.begin(), tokens.end());

    // Cache the tokenized result
    if (dataset->streaming_cache) {
        auto* cache = static_cast<llama_dataset_streaming_cache*>(dataset->streaming_cache);
        cache->put_tokenized(index, int32_tokens);

        // Return the cached data
        void* cached_result = cache->get_tokenized(index);
        if (cached_result) {
            return static_cast<const int32_t*>(cached_result);
        }
    }

    // If caching failed, allocate memory and return (caller must free)
    int32_t* result = static_cast<int32_t*>(malloc(int32_tokens.size() * sizeof(int32_t)));
    if (!result) {
        llama_dataset_set_error("Failed to allocate memory for tokenized sequence");
        return nullptr;
    }

    std::memcpy(result, int32_tokens.data(), int32_tokens.size() * sizeof(int32_t));

    // Store in cached_tensors for cleanup
    if (dataset->cached_tensors && dataset->cached_tensors[index]) {
        dataset->cached_tensors[index]->data = result;
    }

    return result;
}

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

    // Validate format_data before proceeding
    if (!dataset->format_data) {
        llama_dataset_set_error("Parquet format data is null for streaming access");
        return nullptr;
    }

    // Parquet format validation moved to format-specific loader

    // Check if we already have cached streaming data for this sequence
    if (dataset->cached_tensors && dataset->cached_tensors[index] &&
        dataset->cached_tensors[index]->data) {
        return static_cast<const int32_t*>(dataset->cached_tensors[index]->data);
    }

    // Check streaming cache for tokenized sequence
    if (dataset->streaming_cache) {
        auto* cache = static_cast<llama_dataset_streaming_cache*>(dataset->streaming_cache);
        void* cached_tokens = cache->get_tokenized(index);
        if (cached_tokens) {
            return static_cast<const int32_t*>(cached_tokens);
        }
    }

    // Load data on-demand using the Parquet streaming function
    void* streaming_data = nullptr;

    // Parquet streaming data loading moved to format-specific loader

    if (!streaming_data) {
        llama_dataset_set_error("Failed to load Parquet sequence data in streaming mode - no valid data source available");
        return nullptr;
    }

    // Cache the streaming data for future access
    if (dataset->cached_tensors && dataset->cached_tensors[index]) {
        dataset->cached_tensors[index]->data = streaming_data;
    }

    // Also cache in streaming cache if available (performance monitoring integration)
    if (dataset->streaming_cache) {
        auto* cache = static_cast<llama_dataset_streaming_cache*>(dataset->streaming_cache);
        int32_t seq_length = llama_dataset_sequence_length(dataset, index);
        if (seq_length > 0) {
            std::vector<int32_t> tokens(static_cast<const int32_t*>(streaming_data),
                                      static_cast<const int32_t*>(streaming_data) + seq_length);
            cache->put_tokenized(index, tokens);
            // The cache automatically updates performance statistics (hits, misses, memory usage)
        }
    }

    // Optimization manager removed - using registry-based approach (Task H1)

    return static_cast<const int32_t*>(streaming_data);
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

    // Parquet non-streaming tokenization moved to format-specific loader

    // Handle streaming mode using format loader interface (Task H1 - removed hardcoded format logic)
    if (dataset->streaming) {
        // Check if we already have cached data
        if (dataset->cached_tensors && dataset->cached_tensors[index] &&
            dataset->cached_tensors[index]->data) {
            // Return already cached streaming data
            return static_cast<const int32_t*>(dataset->cached_tensors[index]->data);
        }

        // Load streaming data using format loader interface
        // Note: Format loader streaming interface not yet implemented
        // Using generic streaming approach for now
        void* streaming_data = nullptr;
        // streaming_data = dataset->format_loader->get_streaming_data(dataset, index);
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

    // Handle streaming mode using format loader interface (Task H1 - removed hardcoded format logic)
    if (dataset->streaming) {
        // Use generic streaming approach through format loader interface
        // Note: Format loader streaming interface not yet implemented
        // Using GGUF context if available for now
        if (dataset->ctx) {
            // Get tensor size and calculate length
            size_t tensor_size = gguf_get_tensor_size(dataset->ctx, index);
            return tensor_size / sizeof(int32_t);
        } else {
            llama_dataset_set_error("Streaming context is null");
            return 0;
        }
    }

    // Non-streaming path: use cached tensors (Parquet support moved to format-specific loader)

    // Standard non-streaming path: use cached tensors
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
