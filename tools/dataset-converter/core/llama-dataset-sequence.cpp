#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "llama-dataset-gguf.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-utils.h"
#include "llama-dataset.h"
#include "streaming-cache.h"

#ifdef LLAMA_PARQUET
#include "llama-dataset-parquet-internal.h"
#endif

// Helper function to perform on-demand tokenization for streaming mode
static const int32_t* llama_dataset_tokenize_text_streaming(const struct llama_dataset* dataset, uint64_t index, const std::string& text) {
    if (!dataset || !dataset->model) {
        llama_dataset_set_error("Dataset or model is null for tokenization");
        return nullptr;
    }

    // Check if we already have this text tokenized in the streaming cache
    if (dataset->streaming_cache) {
        auto* cache = static_cast<llama_dataset_streaming_cache*>(dataset->streaming_cache);
        void* cached_tokens = cache->get_tokenized(index);
        if (cached_tokens) {
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
        
        ctx = llama_new_context_with_model(dataset->model, ctx_params);
        if (!ctx) {
            llama_dataset_set_error("Failed to create tokenization context");
            return nullptr;
        }
        owns_temp_ctx = true;
    }

    // Tokenize the text
    tokens.resize(text.length() + 16); // Reserve space for tokens
    int n_tokens = llama_tokenize(dataset->model, text.c_str(), text.length(), 
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
    
#ifdef LLAMA_PARQUET
    // Use the existing Parquet streaming function
    extern void * llama_dataset_get_parquet_tensor_data_streaming(const struct llama_dataset * dataset, uint64_t index);
    streaming_data = llama_dataset_get_parquet_tensor_data_streaming(dataset, index);
#endif

    if (!streaming_data) {
        // If direct loading failed, try tokenization if we have a model and text data
        if (dataset->model && dataset->format_data) {
#ifdef LLAMA_PARQUET
            // Try to get text data and tokenize it
            auto* format_data = static_cast<struct parquet_format_data*>(dataset->format_data);
            if (format_data->tokenizer) {
                // This would require implementing text extraction from Parquet
                // For now, we'll return an error but the infrastructure is in place
                llama_dataset_set_error("On-demand tokenization from Parquet text not yet implemented");
                return nullptr;
            }
#endif
        }
        
        llama_dataset_set_error("Failed to load sequence data in streaming mode");
        return nullptr;
    }

    // Cache the streaming data for future access
    if (dataset->cached_tensors && dataset->cached_tensors[index]) {
        dataset->cached_tensors[index]->data = streaming_data;
    }

    // Also cache in streaming cache if available
    if (dataset->streaming_cache) {
        auto* cache = static_cast<llama_dataset_streaming_cache*>(dataset->streaming_cache);
        int32_t seq_length = llama_dataset_sequence_length(dataset, index);
        if (seq_length > 0) {
            std::vector<int32_t> tokens(static_cast<const int32_t*>(streaming_data), 
                                      static_cast<const int32_t*>(streaming_data) + seq_length);
            cache->put_tokenized(index, tokens);
        }
    }

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
