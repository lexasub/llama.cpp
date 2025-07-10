#include "llama-dataset.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <vector>

#include "../../common/log.h"
#include "../../ggml/include/ggml.h"
#include "../../ggml/include/gguf.h"
#include "../../src/llama-impl.h"
#include "llama-dataset-gguf-utils.h"
#include "llama-dataset-gguf.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-parquet.h"
#include "llama-dataset-text.h"
#include "llama-dataset-utils.h"
#include "streaming-optimization-manager.h"

// Factory functions - simple implementations as wrappers
struct llama_dataset * from_gguf(const char * path) {
    return llama_dataset_load_gguf(path, false);
}

struct llama_dataset * from_txt(const char * path, struct llama_model * model) {
    // Use the implementation from llama-dataset-text.cpp
    return llama_dataset_load_text_internal(path, model, false);
}

struct llama_dataset * from_parquet(const char * path) {
    return llama_dataset_load_parquet_internal(path, false);
}

struct llama_dataset * llama_dataset_load_parquet(const char * path, bool streaming) {
    return llama_dataset_load_parquet_internal(path, streaming);
}

bool llama_dataset_save_gguf(struct llama_dataset * dataset, const char * path) {
    if (!dataset || !path) {
        set_error("Invalid parameters for GGUF save");
        return false;
    }

    to_gguf(dataset, path);
    return !llama_dataset_has_error();
}

// Data access functions

/**
 * @brief Get the number of sequences in the dataset.
 *
 * This function returns the number of sequences in the dataset, which is cached
 * during dataset loading for fast access. If the dataset is NULL, it returns 0.
 *
 * @param dataset Dataset to query
 * @return Number of sequences, or 0 if dataset is NULL
 */
uint64_t n_sequences(const struct llama_dataset * dataset) {
    if (!dataset) {
        return 0;
    }

    // First check if we have a cached value
    if (dataset->n_seq > 0) {
        return dataset->n_seq;
    }

    // If not cached, try to get from metadata
    if (dataset->ctx) {
        int32_t key_idx = gguf_find_key(dataset->ctx, DATASET_SEQUENCE_COUNT);
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

// sequence_length is implemented in llama-dataset-sequence.cpp

// sequence is implemented in llama-dataset-sequence.cpp

// Legacy compatibility functions

/**
 * @brief Get the number of sequences in the dataset (legacy function).
 *
 * @param dataset Dataset to query
 * @return Number of sequences, or 0 if dataset is NULL
 */
uint64_t llama_dataset_get_sequence_count(const struct llama_dataset * dataset) {
    return n_sequences(dataset);
}

/**
 * @brief Get the length of a sequence in the dataset (legacy function).
 *
 * @param dataset Dataset to query
 * @param index Index of the sequence
 * @return Length of the sequence, or 0 if dataset is NULL or index is out of bounds
 */
int32_t llama_dataset_get_sequence_length(const struct llama_dataset * dataset, uint64_t index) {
    return sequence_length(dataset, index);
}

/**
 * @brief Get a pointer to the tokens in a sequence (legacy function).
 *
 * @param dataset Dataset to query
 * @param index Index of the sequence
 * @return Pointer to the tokens, or NULL if dataset is NULL or index is out of bounds
 */
const llama_token * llama_dataset_get_sequence(const struct llama_dataset * dataset, uint64_t index) {
    return (const llama_token *)sequence(dataset, index);
}

// sequence_tensor is implemented in llama-dataset-sequence.cpp

// Metadata access functions
const char * dataset_get_metadata_str(const struct llama_dataset * dataset, const char * key) {
    if (!dataset || !dataset->ctx || !key) {
        return nullptr;
    }

    int32_t key_idx = gguf_find_key(dataset->ctx, key);
    if (key_idx < 0) {
        return nullptr;
    }

    enum gguf_type type = gguf_get_kv_type(dataset->ctx, key_idx);
    if (type != GGUF_TYPE_STRING) {
        return nullptr;
    }

    return gguf_get_val_str(dataset->ctx, key_idx);
}

int64_t dataset_get_metadata_int(const struct llama_dataset * dataset, const char * key, int64_t default_value) {
    if (!dataset || !dataset->ctx || !key) {
        return default_value;
    }

    int32_t key_idx = gguf_find_key(dataset->ctx, key);
    if (key_idx < 0) {
        return default_value;
    }

    enum gguf_type type = gguf_get_kv_type(dataset->ctx, key_idx);
    if (type != GGUF_TYPE_INT32 && type != GGUF_TYPE_INT64) {
        return default_value;
    }

    return gguf_get_val_i64(dataset->ctx, key_idx);
}

float dataset_get_metadata_float(const struct llama_dataset * dataset, const char * key, float default_value) {
    if (!dataset || !dataset->ctx || !key) {
        return default_value;
    }

    int32_t key_idx = gguf_find_key(dataset->ctx, key);
    if (key_idx < 0) {
        return default_value;
    }

    enum gguf_type type = gguf_get_kv_type(dataset->ctx, key_idx);
    if (type != GGUF_TYPE_FLOAT32) {
        return default_value;
    }

    return gguf_get_val_f32(dataset->ctx, key_idx);
}

// Conversion utility
void to_gguf(struct llama_dataset * dataset, const char * path) {
    if (!dataset || !path) {
        set_error("Invalid parameters for GGUF conversion");
        return;
    }

    // For GGUF datasets, we can use the existing GGUF context
    if (dataset->type == DATASET_GGUF && dataset->ctx) {
        // If the dataset is in streaming mode, we need to ensure all tensors are loaded
        if (dataset->streaming) {
            LLAMA_LOG_INFO("Converting streaming GGUF dataset to file: %s", path);

            // For streaming datasets, we need to load all tensor data
            uint64_t n_seq = n_sequences(dataset);
            for (uint64_t i = 0; i < n_seq; i++) {
                // This will trigger loading the tensor data if not already loaded
                const int32_t * tokens = sequence(dataset, i);
                if (!tokens) {
                    set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to load tensor data for streaming conversion");
                    return;
                }
            }
        }

        // Write GGUF file manually since we need to handle streaming data properly
        FILE* file = fopen(path, "wb");
        if (!file) {
            set_error_with_code(DATASET_ERROR_FILE_NOT_FOUND, "Failed to open output file for writing");
            return;
        }

        // Get the meta data size and write header + metadata
        size_t meta_size = gguf_get_meta_size(dataset->ctx);
        void* meta_data = malloc(meta_size);
        if (!meta_data) {
            fclose(file);
            set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate memory for GGUF metadata");
            return;
        }

        // Get and write the metadata
        gguf_get_meta_data(dataset->ctx, meta_data);
        if (fwrite(meta_data, 1, meta_size, file) != meta_size) {
            free(meta_data);
            fclose(file);
            set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to write GGUF metadata");
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
                set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Tensor data not available for writing");
                return;
            }

            if (tensor_size == 0) {
                fclose(file);
                set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Invalid tensor size for writing");
                return;
            }

            // Write tensor data with proper alignment
            if (fwrite(tensor_data, 1, tensor_size, file) != tensor_size) {
                fclose(file);
                set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to write tensor data");
                return;
            }

            // Add padding to align to 32-byte boundary if needed
            size_t padding = (32 - (tensor_size % 32)) % 32;
            if (padding > 0) {
                char zero_padding[32] = {0};
                if (fwrite(zero_padding, 1, padding, file) != padding) {
                    fclose(file);
                    set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to write tensor padding");
                    return;
                }
            }
        }

        fclose(file);
        LLAMA_LOG_INFO("Successfully wrote GGUF dataset to %s", path);
        return;
    }

    // For other formats (TEXT, PARQUET), we need to create a new GGUF file
    LLAMA_LOG_INFO("Converting %s dataset to GGUF file: %s",
                  dataset->type == DATASET_TEXT ? "TEXT" : "PARQUET", path);

    // Create a new GGUF context
    struct gguf_context * new_ctx = gguf_init_empty();
    if (!new_ctx) {
        set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGUF context for conversion");
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
                case GGUF_TYPE_INT64:
                    gguf_set_val_i64(new_ctx, key, gguf_get_val_i64(dataset->ctx, i));
                    break;
                case GGUF_TYPE_FLOAT32:
                    gguf_set_val_f32(new_ctx, key, gguf_get_val_f32(dataset->ctx, i));
                    break;
                default:
                    // Skip other types for now
                    LLAMA_LOG_WARN("Skipping metadata key '%s' with unsupported type %d", key, type);
                    break;
            }
        }
    }

    // Add or update standard metadata
    gguf_set_val_str(new_ctx, DATASET_SOURCE_FORMAT,
                    dataset->type == DATASET_TEXT ? "text" :
                    dataset->type == DATASET_PARQUET ? "parquet" : "gguf");

    // Add creation timestamp
    time_t now = time(NULL);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    gguf_set_val_str(new_ctx, DATASET_CREATION_TIME, timestamp);

    // Set sequence count
    uint64_t seq_count = n_sequences(dataset);
    gguf_set_val_i32(new_ctx, DATASET_SEQUENCE_COUNT, (int32_t)seq_count);

    // Find maximum sequence length
    int32_t max_length = 0;
    for (uint64_t i = 0; i < seq_count; i++) {
        int32_t len = sequence_length(dataset, i);
        if (len > max_length) {
            max_length = len;
        }
    }
    gguf_set_val_u32(new_ctx, DATASET_MAX_LENGTH, (uint32_t)max_length);

    // Add additional useful metadata
    gguf_set_val_str(new_ctx, "dataset.version", "1.0");
    gguf_set_val_str(new_ctx, "dataset.format", "llama-dataset");

    // Add total token count
    uint64_t total_tokens = 0;
    for (uint64_t i = 0; i < seq_count; i++) {
        total_tokens += sequence_length(dataset, i);
    }
    gguf_set_val_u32(new_ctx, "dataset.total_tokens", (uint32_t)total_tokens);

    // Create GGML context for tensor data
    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ 128ull*1024ull*1024ull,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context* ggml_ctx = ggml_init(ggml_params);
    if (!ggml_ctx) {
        gguf_free(new_ctx);
        set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGML context for conversion");
        return;
    }

    // Add all sequences as tensors
    for (uint64_t i = 0; i < seq_count; i++) {
        // Get sequence data
        const int32_t * tokens = sequence(dataset, i);
        int32_t length = sequence_length(dataset, i);

        if (!tokens || length <= 0) {
            LLAMA_LOG_WARN("Skipping invalid sequence at index %zu", i);
            continue;
        }

        // Create tensor name (use format "seq_XXXXX" with zero-padding)
        char tensor_name[32];
        snprintf(tensor_name, sizeof(tensor_name), "seq_%05" PRIu64, i);

        // Create tensor in GGML context
        int64_t ne[1] = { (int64_t)length };
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
        set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to write GGUF file");
        return;
    }

    // Clean up
    ggml_free(ggml_ctx);
    gguf_free(new_ctx);

    LLAMA_LOG_INFO("Successfully converted dataset to GGUF file: %s", path);
}

// Cleanup
void llama_dataset_free(struct llama_dataset * dataset) {
    if (!dataset) {
        return;
    }

    // Free streaming optimization manager if it exists
    if (dataset->optimization_manager) {
        delete static_cast<StreamingOptimizationManager*>(dataset->optimization_manager);
        dataset->optimization_manager = nullptr;
    }

    // Free streaming cache if in streaming mode
    if (dataset->streaming && dataset->streaming_cache) {
        delete static_cast<StreamingCache*>(dataset->streaming_cache);
        dataset->streaming_cache = nullptr;
    }

    // Free cached tensor pointers and their data if in streaming mode
    if (dataset->cached_tensors) {
        // In streaming mode, we need to free the tensor data that we allocated
        if (dataset->streaming) {
            uint64_t n_seq = n_sequences(dataset);
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
