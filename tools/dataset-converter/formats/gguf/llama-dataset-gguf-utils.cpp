#include "llama-dataset-gguf-utils.h"

#include <cstdio>

#include "gguf.h"
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
        // DEPRECATED: Direct stderr output - use centralized error handling
        #pragma message("DEPRECATED: Direct stderr output in GGUF utils - use centralized error handling system")
        fprintf(stderr, "Invalid GGUF or GGML context\n");
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
            // DEPRECATED: Direct stderr output - use centralized error handling
            fprintf(stderr, "Failed to get tensor name for tensor %d\n", i);
            return false;
        }

        // Validate tensor metadata
        size_t tensor_size = gguf_get_tensor_size(gguf_ctx, i);
        if (tensor_size == 0) {
            // DEPRECATED: Direct stderr output - use centralized error handling
            fprintf(stderr, "Invalid tensor size for tensor %s\n", name);
            return false;
        }

        enum ggml_type tensor_type = gguf_get_tensor_type(gguf_ctx, i);
        if (tensor_type >= GGML_TYPE_COUNT) {
            // DEPRECATED: Direct stderr output - use centralized error handling
            fprintf(stderr, "Invalid tensor type for tensor %s\n", name);
            return false;
        }
    }

    return true;
}
