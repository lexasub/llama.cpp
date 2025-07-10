#include "llama-dataset-gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../../common/log.h"
#include "llama-dataset-gguf-utils.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-utils.h"
#include "llama-dataset-validation.h"
#include "llama-impl.h"

// Validate GGUF file format and detect corruption
static bool validate_gguf_file_format(const char* path) {
    FILE* file = fopen(path, "rb");
    if (!file) {
        return false;
    }

    // Read and validate GGUF magic number
    char magic[4];
    if (fread(magic, 1, 4, file) != 4) {
        fclose(file);
        return false;
    }

    // Check GGUF magic bytes
    if (memcmp(magic, "GGUF", 4) != 0) {
        fclose(file);
        return false;
    }

    // Read version
    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, file) != 1) {
        fclose(file);
        return false;
    }

    // Validate version (currently support versions 1, 2, 3)
    if (version < 1 || version > 3) {
        fclose(file);
        return false;
    }

    // Read tensor count and kv count
    uint64_t tensor_count, kv_count;
    if (fread(&tensor_count, sizeof(uint64_t), 1, file) != 1 ||
        fread(&kv_count, sizeof(uint64_t), 1, file) != 1) {
        fclose(file);
        return false;
    }

    // Basic sanity checks
    if (tensor_count > 1000000 || kv_count > 1000000) {  // Reasonable limits
        fclose(file);
        return false;
    }

    // Get file size for additional validation
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fclose(file);

    // File should be at least header size + some data
    if (file_size < 32) {
        return false;
    }

    return true;
}

// Load a dataset from a GGUF file with streaming option
struct llama_dataset* llama_dataset_load_gguf(const char* path, bool streaming) {
    if (!path) {
        set_error("GGUF path is null");
        return nullptr;
    }

    // Check if file exists
    FILE* file = fopen(path, "rb");
    if (!file) {
        set_error_with_code(DATASET_ERROR_FILE_NOT_FOUND, "GGUF file not found");
        return nullptr;
    }
    fclose(file);

    // Validate GGUF file format and detect corruption using comprehensive validation
    struct validation_result validation;
    if (!validate_gguf_file(path, &validation)) {
        set_error_with_code(validation.error_code, validation.error_message);
        return nullptr;
    }

    // Create dataset structure
    struct llama_dataset* dataset = dataset_alloc(DATASET_GGUF, streaming);
    if (!dataset) {
        // Error already set by dataset_alloc
        return nullptr;
    }

    // In streaming mode, we store the file path for later use
    if (streaming) {
        // Load GGUF context without tensor data
        struct gguf_init_params params;
        params.no_alloc = true;
        params.ctx = nullptr;

        dataset->ctx = gguf_init_from_file(path, params);
        if (!dataset->ctx) {
            set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to load GGUF file");
            free(dataset);
            return nullptr;
        }

        dataset->format_data = strdup(path);
        if (!dataset->format_data) {
            set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate path storage");
            gguf_free(dataset->ctx);
            free(dataset);
            return nullptr;
        }
    } else {
        // In non-streaming mode, we load all tensor data into memory
        struct gguf_init_params params;
        params.no_alloc = false;
        params.ctx = &dataset->ggml_ctx;

        dataset->ctx = gguf_init_from_file(path, params);
        if (!dataset->ctx) {
            set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to load GGUF file");
            free(dataset);
            return nullptr;
        }

        // The GGML context is automatically created and tensors loaded by gguf_init_from_file
        if (!dataset->ggml_ctx) {
            set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGML context");
            gguf_free(dataset->ctx);
            free(dataset);
            return nullptr;
        }

        // Validate that tensors were loaded correctly
        if (!gguf_load_tensors(dataset->ctx, dataset->ggml_ctx)) {
            set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to validate GGUF tensors");
            ggml_free(dataset->ggml_ctx);
            gguf_free(dataset->ctx);
            free(dataset);
            return nullptr;
        }
    }

    // Cache tensor pointers for fast access
    if (!dataset_cache_tensors(dataset)) {
        // Error already set by dataset_cache_tensors
        if (dataset->ggml_ctx) {
            ggml_free(dataset->ggml_ctx);
        }
        gguf_free(dataset->ctx);
        if (dataset->format_data) {
            free(dataset->format_data);
        }
        free(dataset);
        return nullptr;
    }

    return dataset;
}

// Get tensor data from a GGUF file in streaming mode
void* gguf_get_tensor_data_streaming(const struct llama_dataset* dataset, uint64_t index) {
    if (!dataset || !dataset->ctx || !dataset->streaming || !dataset->format_data) {
        set_error("Invalid dataset for streaming data access");
        return nullptr;
    }

    // Check streaming cache first
    if (dataset->streaming_cache) {
        StreamingCache* cache = static_cast<StreamingCache*>(dataset->streaming_cache);
        void* cached_data = cache->get(index);
        if (cached_data) {
            LLAMA_LOG_DEBUG("Retrieved sequence %zu from streaming cache", index);
            return cached_data;
        }
    }

    // Get tensor info from GGUF context
    const char* name = gguf_get_tensor_name(dataset->ctx, index);
    if (!name) {
        set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Invalid tensor index");
        return nullptr;
    }

    // Get tensor metadata
    int64_t tensor_id = gguf_find_tensor(dataset->ctx, name);
    if (tensor_id < 0) {
        set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Tensor not found in GGUF context");
        return nullptr;
    }

    // Get tensor size and offset
    size_t tensor_size = gguf_get_tensor_size(dataset->ctx, tensor_id);
    size_t tensor_offset = gguf_get_tensor_offset(dataset->ctx, tensor_id);
    size_t data_offset = gguf_get_data_offset(dataset->ctx);
    size_t file_offset = data_offset + tensor_offset;

    if (tensor_size == 0) {
        set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Invalid tensor size");
        return nullptr;
    }

    // Allocate memory for tensor data
    void* data = malloc(tensor_size);
    if (!data) {
        set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate tensor data");
        return nullptr;
    }

    // Open file and seek to tensor data
    FILE* file = fopen((const char*)dataset->format_data, "rb");
    if (!file) {
        set_error_with_code(DATASET_ERROR_FILE_NOT_FOUND, "Failed to open GGUF file for streaming");
        free(data);
        return nullptr;
    }

    // Seek to tensor data
    if (fseek(file, (long)file_offset, SEEK_SET) != 0) {
        set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to seek to tensor data");
        fclose(file);
        free(data);
        return nullptr;
    }

    // Read tensor data
    if (fread(data, 1, tensor_size, file) != tensor_size) {
        set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to read tensor data");
        fclose(file);
        free(data);
        return nullptr;
    }

    fclose(file);

    // Add to streaming cache (cache takes ownership of the data)
    if (dataset->streaming_cache) {
        StreamingCache* cache = static_cast<StreamingCache*>(dataset->streaming_cache);

        // Make a copy for the cache since the cache will manage the memory
        void* cache_data = malloc(tensor_size);
        if (cache_data) {
            memcpy(cache_data, data, tensor_size);
            cache->put(index, cache_data, tensor_size);
            LLAMA_LOG_DEBUG("Added sequence %zu to streaming cache (size: %zu bytes)", index, tensor_size);
        }
    }

    return data;
}
