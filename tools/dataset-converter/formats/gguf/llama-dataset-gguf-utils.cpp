#include "llama-dataset-gguf-utils.h"

#include <cstdio>

#include "gguf.h"
#include "../../core/llama-dataset.h"  // For error handling functions
size_t llama_dataset_gguf_get_data_size(const struct gguf_context * ctx);
size_t llama_dataset_gguf_get_data_size(const struct gguf_context * ctx) {
    if (!ctx) {
        return 0;
    }

    size_t total_size = 0;
    const int n_tensors = gguf_get_n_tensors(ctx);

    for (int i = 0; i < n_tensors; ++i) {
        size_t tensor_size = gguf_get_tensor_size(ctx, i);
        // Add alignment padding
        size_t alignment = gguf_get_alignment(ctx);
        if (alignment > 0) {
            tensor_size = (tensor_size + alignment - 1) & ~(alignment - 1);
        }
        total_size += tensor_size;
    }

    return total_size;
}

bool llama_dataset_gguf_load_tensors(const struct gguf_context * gguf_ctx, struct ggml_context * ggml_ctx) {
    if (!gguf_ctx || !ggml_ctx) {
        // Use centralized error handling instead of direct stderr output
        llama_dataset_set_error("Invalid GGUF or GGML context");
        return false;
    }

    // This function is a compatibility wrapper
    // In the actual GGUF API, tensor loading is handled by gguf_init_from_file
    // when called with appropriate parameters

    // Since we're implementing this as a utility function, we'll validate
    // that the tensors can be accessed and return true if everything looks good

    const int n_tensors = gguf_get_n_tensors(gguf_ctx);

    if (n_tensors <= 0) {
        return true; // No tensors to load is considered success
    }

    // Validate that we can access tensor metadata
    for (int i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        if (!name) {
            // Use centralized error handling instead of direct stderr output
            llama_dataset_set_error("Failed to get tensor name for tensor");
            return false;
        }

        // Validate tensor metadata
        size_t tensor_size = gguf_get_tensor_size(gguf_ctx, i);
        if (tensor_size == 0) {
            // Use centralized error handling instead of direct stderr output
            llama_dataset_set_error("Invalid tensor size for tensor");
            return false;
        }

        enum ggml_type tensor_type = gguf_get_tensor_type(gguf_ctx, i);
        if (tensor_type >= GGML_TYPE_COUNT) {
            // Use centralized error handling instead of direct stderr output
            llama_dataset_set_error("Invalid tensor type for tensor");
            return false;
        }
    }

    return true;
}
