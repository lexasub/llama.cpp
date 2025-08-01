#include "llama-dataset.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>

#include "common.h"
#include "ggml/include/ggml.h"
#include "ggml/include/gguf.h"
#include "llama-dataset-gguf-utils.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-parquet.h"
#include "llama-dataset-text.h"
#include "llama-dataset-utils.h"
#include "src/llama-impl.h"
#include "streaming-optimization-manager.h"

// Factory functions - simple implementations as wrappers
struct llama_dataset * llama_dataset_from_gguf(const common_params * params) {
    return llama_dataset_load_gguf(params);
}

struct llama_dataset * llama_dataset_from_txt(const common_params * params, struct llama_model * model) {
    return llama_dataset_load_text_internal(params, model);
}

#ifdef LLAMA_DATASET_PARQUET_SUPPORT
struct llama_dataset * llama_dataset_from_parquet(const common_params * params) {
    return llama_dataset_load_parquet_internal(params);
}
#endif

/**
 * @brief Get the number of sequences in the dataset.
 *
 * This function returns the number of sequences in the dataset, which is cached
 * during dataset loading for fast access. If the dataset is NULL, it returns 0.
 *
 * @param dataset Dataset to query
 * @return Number of sequences, or 0 if dataset is NULL
 */
uint64_t llama_dataset_n_sequences(const struct llama_dataset * dataset) {
    if (!dataset) {
        return 0;
    }

    // First check if we have a cached value
    if (dataset->n_seq > 0) {
        return dataset->n_seq;
    }

    // If not cached, try to get from metadata
    if (dataset->ctx) {
        int32_t key_idx = gguf_find_key(dataset->ctx, TRAINING_SEQUENCE_COUNT);
        if (key_idx >= 0) {
            enum gguf_type type = gguf_get_kv_type(dataset->ctx, key_idx);
            if (type == GGUF_TYPE_INT32 || type == GGUF_TYPE_INT64) {
                return (uint64_t)gguf_get_val_i64(dataset->ctx, key_idx);
            }
        }

        // If not in metadata, count tensors
        return gguf_get_n_tensors(dataset->ctx);
    }

    return 0;
}

// Metadata access functions
const char * llama_dataset_get_metadata_str(const struct llama_dataset * dataset, const char * key) {
    if (!dataset || !dataset->ctx || !key) {
        return nullptr;
    }

    int32_t key_idx = gguf_find_key(dataset->ctx, key);
    if (key_idx < 0) {
        return nullptr;
    }

    enum gguf_type type = gguf_get_kv_type(dataset->ctx, key_idx);
    return type == GGUF_TYPE_STRING ? gguf_get_val_str(dataset->ctx, key_idx) : nullptr;
}

int64_t llama_dataset_get_metadata_int(const struct llama_dataset * dataset, const char * key, int64_t default_value) {
    if (!dataset || !dataset->ctx || !key) {
        return default_value;
    }

    int32_t key_idx = gguf_find_key(dataset->ctx, key);
    if (key_idx < 0) {
        return default_value;
    }

    enum gguf_type type = gguf_get_kv_type(dataset->ctx, key_idx);
    return type == GGUF_TYPE_INT32 || type == GGUF_TYPE_INT64 ? gguf_get_val_i64(dataset->ctx, key_idx) : default_value;
}

float llama_dataset_get_metadata_float(const struct llama_dataset * dataset, const char * key, float default_value) {
    if (!dataset || !dataset->ctx || !key) {
        return default_value;
    }

    int32_t key_idx = gguf_find_key(dataset->ctx, key);
    if (key_idx < 0) {
        return default_value;
    }

    enum gguf_type type = gguf_get_kv_type(dataset->ctx, key_idx);
    return type == GGUF_TYPE_FLOAT32 ? gguf_get_val_f32(dataset->ctx, key_idx) : default_value;
}

// Conversion utility
void llama_dataset_to_gguf(struct llama_dataset * dataset, const char * path) {
    if (!dataset || !path) {
        llama_dataset_set_error("Invalid parameters for GGUF conversion\n");
        return;
    }

    // For GGUF datasets, we can use the existing GGUF context
    if (dataset->type == DATASET_GGUF && dataset->ctx) {
        // If the dataset is in streaming mode, we need to ensure all tensors are loaded
        if (dataset->streaming) {
            LLAMA_LOG_INFO("Converting streaming GGUF dataset to file: %s\n", path);

            // For streaming datasets, we need to load all tensor data
            uint64_t n_seq = llama_dataset_n_sequences(dataset);
            for (uint64_t i = 0; i < n_seq; i++) {
                // This will trigger loading the tensor data if not already loaded
                if (llama_dataset_sequence(dataset, i) == nullptr) {
                    llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to load tensor data for streaming conversion");
                    return;
                }
            }
        }

        // Write GGUF file manually since we need to handle streaming data properly
        FILE* file = fopen(path, "wb");
        if (!file) {
            llama_dataset_set_error_with_code(DATASET_ERROR_FILE_NOT_FOUND, "Failed to open output file for writing");
            return;
        }

        // Get the meta data size and write header + metadata
        size_t meta_size = gguf_get_meta_size(dataset->ctx);
        void* meta_data = malloc(meta_size);
        if (!meta_data) {
            fclose(file);
            llama_dataset_set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate memory for GGUF metadata");
            return;
        }

        // Get and write the metadata
        gguf_get_meta_data(dataset->ctx, meta_data);
        if (fwrite(meta_data, 1, meta_size, file) != meta_size) {
            free(meta_data);
            fclose(file);
            llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to write GGUF metadata");
            return;
        }
        free(meta_data);

        // Write tensor data - for streaming mode, we need to get data from cache
        uint64_t n_tensors = gguf_get_n_tensors(dataset->ctx);
        for (uint64_t i = 0; i < n_tensors; i++) {
            const void* tensor_data = nullptr;
            size_t tensor_size = 0;

            if (dataset->cached_tensors && dataset->cached_tensors[i]) {
                if (dataset->cached_tensors[i]->data) {
                    tensor_data = dataset->cached_tensors[i]->data;
                    tensor_size = ggml_nbytes(dataset->cached_tensors[i]);
                } else {
                    // Get tensor size from GGUF context
                    tensor_size = gguf_get_tensor_size(dataset->ctx, i);
                }
            } else {
                tensor_size = gguf_get_tensor_size(dataset->ctx, i);
            }

            if (!tensor_data) {
                fclose(file);
                llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Tensor data not available for writing");
                return;
            }

            if (tensor_size == 0) {
                fclose(file);
                llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Invalid tensor size for writing");
                return;
            }

            // Write tensor data with proper alignment
            if (fwrite(tensor_data, 1, tensor_size, file) != tensor_size) {
                fclose(file);
                llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to write tensor data");
                return;
            }

            // Add padding to align to 32-byte boundary if needed
            size_t padding = (32 - (tensor_size % 32)) % 32;
            if (padding > 0) {
                char zero_padding[32] = {0};
                if (fwrite(zero_padding, 1, padding, file) != padding) {
                    fclose(file);
                    llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to write tensor padding");
                    return;
                }
            }
        }

        fclose(file);
        LLAMA_LOG_INFO("Successfully wrote GGUF dataset to %s\n", path);
        return;
    }

    // For other formats (TEXT, PARQUET), we need to create a new GGUF file
    LLAMA_LOG_INFO("Converting %s dataset to GGUF file: %s\n", dataset->type == DATASET_TEXT ? "TEXT" : "PARQUET", path);

    // Create a new GGUF context
    struct gguf_context * new_ctx = gguf_init_empty();
    if (!new_ctx) {
        llama_dataset_set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGUF context for conversion");
        return;
    }

    // Copy metadata from the original dataset
    if (dataset->ctx) {
        // Copy all key-value pairs
        int n_kv = gguf_get_n_kv(dataset->ctx);
        for (int i = 0; i < n_kv; i++) {
            const char* key = gguf_get_key(dataset->ctx, i);
            enum gguf_type type = gguf_get_kv_type(dataset->ctx, i);

            switch (type) {
                case GGUF_TYPE_STRING:
                    gguf_set_val_str(new_ctx, key, gguf_get_val_str(dataset->ctx, i));
                    break;
                case GGUF_TYPE_INT32:
                    gguf_set_val_i32(new_ctx, key, gguf_get_val_i32(dataset->ctx, i));
                    break;
                case GGUF_TYPE_UINT32:
                    gguf_set_val_u32(new_ctx, key, gguf_get_val_u32(dataset->ctx, i));
                    break;
                case GGUF_TYPE_INT64:
                    gguf_set_val_i64(new_ctx, key, gguf_get_val_i64(dataset->ctx, i));
                    break;
                case GGUF_TYPE_UINT64:
                    gguf_set_val_u64(new_ctx, key, gguf_get_val_u64(dataset->ctx, i));
                    break;
                case GGUF_TYPE_FLOAT32:
                    gguf_set_val_f32(new_ctx, key, gguf_get_val_f32(dataset->ctx, i));
                    break;
                case GGUF_TYPE_FLOAT64:
                    gguf_set_val_f64(new_ctx, key, gguf_get_val_f64(dataset->ctx, i));
                    break;
                default:
                    LLAMA_LOG_WARN("Bad metadata key '%s' with type %d", key, type);
                    break;
            }
        }
    }

    // Add or update standard metadata
    gguf_set_val_str(new_ctx, TRAINING_FORMAT_SOURCE, dataset->type == DATASET_TEXT ? "text" : dataset->type == DATASET_PARQUET ? "parquet" : "gguf");

    // Add creation timestamp
    time_t now = time(nullptr);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    gguf_set_val_str(new_ctx, TRAINING_CREATION_TIME, timestamp);

    // Set sequence count
    uint64_t seq_count = llama_dataset_n_sequences(dataset);
    gguf_set_val_i32(new_ctx, TRAINING_SEQUENCE_COUNT, static_cast<int32_t>(seq_count));

    // Find maximum sequence length
    int32_t max_length = 0;
    for (uint64_t i = 0; i < seq_count; i++) {
        int32_t len = llama_dataset_sequence_length(dataset, i);
        if (len > max_length) {
            max_length = len;
        }
    }
    gguf_set_val_u32(new_ctx, TRAINING_MAX_LENGTH, static_cast<uint32_t>(max_length));

    // Add additional useful metadata
    gguf_set_val_i16(new_ctx, TRAINING_FORMAT_VERSION, 1000);
    gguf_set_val_str(new_ctx, TRAINING_DATASET_NAME, "llama-dataset");
    //TODO fill other
    /*
     *training.dataset.source: string (optional) - URL or description of the data source.
     *training.tokenizer.gguf.model: string - Tokenizer model name (llama, gpt2, etc.).
     *training.tokenizer.gguf.vocab: array[string] - Tokenizer dictionary.
     *training.tokenizer.gguf.merges: array[string] - Tokenizer merges (for BPE).
     *training.tokenizer.gguf.pre: string (optional) - Pre-tokenization architecture.
     */


    // Add total token count
    uint64_t total_tokens = 0;
    for (uint64_t i = 0; i < seq_count; i++) {
        total_tokens += llama_dataset_sequence_length(dataset, i);
    }
    gguf_set_val_u64(new_ctx, TRAINING_SEQUENCE_COUNT, static_cast<uint32_t>(total_tokens));

    // Create GGML context for tensor data
    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ 128ull*1024ull*1024ull,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context* ggml_ctx = ggml_init(ggml_params);
    if (!ggml_ctx) {
        gguf_free(new_ctx);
        llama_dataset_set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGML context for conversion");
        return;
    }

    // Add all sequences as tensors
    for (uint64_t i = 0; i < seq_count; i++) {
        // Get sequence data
        const int32_t * tokens = llama_dataset_sequence(dataset, i);
        int32_t length = llama_dataset_sequence_length(dataset, i);

        if (!tokens || length <= 0) {
            LLAMA_LOG_WARN("Skipping invalid sequence at index %zu\n", i);
            continue;
        }

        // Create tensor name (use format "seq_XXXXX" with zero-padding)
        char tensor_name[32];
        snprintf(tensor_name, sizeof(tensor_name), "seq_%05" PRIu64, i);

        // Create tensor in GGML context
        int64_t ne[1] = { static_cast<int64_t>(length) };
        struct ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, GGML_TYPE_I32, 1, ne);
        ggml_set_name(tensor, tensor_name);

        // Copy data to tensor
        memcpy(tensor->data, tokens, length * sizeof(int32_t));

        // Add tensor to GGUF context
        gguf_add_tensor(new_ctx, tensor);
    }

    // Use the built-in GGUF writing functionality
    if (!gguf_write_to_file(new_ctx, path, false)) {
        ggml_free(ggml_ctx);
        gguf_free(new_ctx);
        llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to write GGUF file");
        return;
    }

    // Clean up
    ggml_free(ggml_ctx);
    gguf_free(new_ctx);

    LLAMA_LOG_INFO("Successfully converted dataset to GGUF file: %s\n", path);
}

// Cleanup
void llama_dataset_free(struct llama_dataset * dataset) {
    if (!dataset) {
        return;
    }

    // Free streaming optimization manager if it exists
    if (dataset->optimization_manager) {
        delete static_cast<llama_dataset_stream_optimization_manager *>(dataset->optimization_manager);
        dataset->optimization_manager = nullptr;
    }

    // Free streaming cache if in streaming mode
    if (dataset->streaming && dataset->streaming_cache) {
        delete static_cast<llama_dataset_streaming_cache *>(dataset->streaming_cache);
        dataset->streaming_cache = nullptr;
    }

    // Free cached tensor pointers and their data if in streaming mode
    if (dataset->cached_tensors) {
        // In streaming mode, we need to free the tensor data that we allocated
        if (dataset->streaming) {
            uint64_t n_seq = llama_dataset_n_sequences(dataset);
            for (uint64_t i = 0; i < n_seq; i++) {
                if (dataset->cached_tensors[i]) {
                    // In streaming mode, tensor data is now managed by the streaming cache
                    // so we only need to free the placeholder tensor structure
                    free(dataset->cached_tensors[i]);
                    dataset->cached_tensors[i] = nullptr;
                }
            }
        }

        free(dataset->cached_tensors);
        dataset->cached_tensors = nullptr;
    }

    // Free GGML context
    if (dataset->ggml_ctx) {
        ggml_free(dataset->ggml_ctx);
        dataset->ggml_ctx = nullptr;
    }

    // Free GGUF context
    if (dataset->ctx) {
        gguf_free(dataset->ctx);
        dataset->ctx = nullptr;
    }

    // Free format-specific data based on type
    if (dataset->format_data) {
        switch (dataset->type) {
            case DATASET_TEXT:
                // Clean up text-specific resources
                // For example, if format_data contains tokenizer state or file handles
                break;

            case DATASET_PARQUET:
                // Clean up Parquet-specific resources
                // For example, if format_data contains Arrow/Parquet objects
#ifdef LLAMA_DATASET_PARQUET_SUPPORT
                // Parquet-specific cleanup code would go here
#endif
                break;

            case DATASET_GGUF:
                // For GGUF in streaming mode, format_data contains the file path
                // which is a simple malloc'd string
                free(dataset->format_data);
                break;
        }

        dataset->format_data = nullptr;
    }

    // Finally free the dataset structure itself
    free(dataset);

    // Note: We don't clear the error state here because the caller might want to check
    // for errors after freeing the dataset
}
