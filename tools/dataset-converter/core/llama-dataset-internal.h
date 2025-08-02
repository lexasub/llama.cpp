#pragma once

/**
 * @file llama-dataset-internal.h
 * @brief Internal dataset structure definition.
 *
 * This header contains the internal structure definition for llama_dataset.
 * It should only be included by implementation files (.cpp) in the dataset-converter module.
 * The public API in llama-dataset.h treats llama_dataset as an opaque structure.
 */

#include "llama-dataset.h"
#include "ggml/include/gguf.h"
#include "ggml/include/ggml.h"
#include "llama.h"

#ifdef __cplusplus
#include <string>
#endif

// Forward declarations for Windows compatibility
#ifdef __cplusplus
class llama_dataset_streaming_cache;
extern "C" {
#endif

/**
 * @brief Core data structure for dataset representation.
 *
 * This structure contains all the necessary data for representing a dataset,
 * including the GGUF context, GGML context, and cached tensor pointers.
 *
 * This structure is internal and should not be exposed to the public API.
 */
struct llama_dataset {
    struct gguf_context * ctx;              // Universal GGUF representation
    struct ggml_context * ggml_ctx;         // Tensor storage (when not streaming)
    struct ggml_tensor ** cached_tensors;   // Fast access cache (vector-like)
    uint64_t n_seq;                         // Cached sequence count
    enum dataset_type type;                 // Format type
    bool streaming;                         // Streaming mode flag
    void * format_data;                     // Format-specific state
    
    // NEW: Tokenization support
    struct llama_model * model;             // Llama model for tokenization
    struct llama_context * tokenizer_ctx;  // Tokenizer context
    bool owns_model;                        // Whether dataset owns the model
    
#ifdef __cplusplus
    llama_dataset_streaming_cache * streaming_cache;       // LRU cache for streaming data
    void * optimization_manager;            // Streaming optimization manager
    std::string                             column;
#else
    void * streaming_cache;                 // Opaque pointer for C compatibility
    void * optimization_manager;            // Opaque pointer for optimization manager
#endif
};

#ifdef __cplusplus
}
#endif
