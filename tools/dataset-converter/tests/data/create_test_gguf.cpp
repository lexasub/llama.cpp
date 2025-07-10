#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include "ggml/include/ggml.h"
#include "ggml/include/gguf.h"
#include "llama-dataset-gguf-utils.h"

// Simple utility to create a test GGUF dataset file
int main() {
    const char* output_file = "test_data/small_dataset.gguf";

    // Create a new GGUF context
    struct gguf_context* ctx = gguf_init_empty();
    if (!ctx) {
        std::cerr << "Failed to create GGUF context" << std::endl;
        return 1;
    }

    // Create GGML context for tensor data
    struct ggml_init_params params = {
        /*.mem_size   =*/ 128ull*1024ull*1024ull,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context* ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        std::cerr << "Failed to create GGML context" << std::endl;
        gguf_free(ctx);
        return 1;
    }

    // Add metadata
    gguf_set_val_str(ctx, "dataset.source", "test");
    gguf_set_val_u64(ctx, "dataset.n_sequences", 3);
    gguf_set_val_u32(ctx, "dataset.max_length", 10);

    // Create some test sequences
    std::vector<std::vector<int32_t>> sequences = {
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25, 26, 27, 28, 29, 30}
    };

    // Add sequences as tensors
    for (size_t i = 0; i < sequences.size(); i++) {
        char tensor_name[32];
        snprintf(tensor_name, sizeof(tensor_name), "seq_%05zu", i);

        // Create tensor in GGML context
        int64_t ne[1] = { (int64_t)sequences[i].size() };
        struct ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, GGML_TYPE_I32, 1, ne);
        ggml_set_name(tensor, tensor_name);

        // Copy data to tensor
        std::memcpy(tensor->data, sequences[i].data(), sequences[i].size() * sizeof(int32_t));

        // Add tensor to GGUF context
        gguf_add_tensor(ctx, tensor);
    }

    // Write to file using the standard GGUF API
    if (!gguf_write_to_file(ctx, output_file, false)) {
        std::cerr << "Failed to write GGUF file" << std::endl;
        ggml_free(ggml_ctx);
        gguf_free(ctx);
        return 1;
    }

    ggml_free(ggml_ctx);
    gguf_free(ctx);

    std::cout << "Created test GGUF dataset: " << output_file << std::endl;
    return 0;
}
