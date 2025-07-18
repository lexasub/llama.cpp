#include "llama-dataset-gguf.h"

#include <cstdio>
#include <cstring>
#include <string>

#include "common.h"
#include "common/log.h"
#include "llama-dataset-gguf-utils.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-utils.h"
#include "llama-dataset-validation.h"
#include "llama-impl.h"

// Load a dataset from a GGUF file with streaming option
struct llama_dataset* llama_dataset_load_gguf(const common_params * common_params) {
    if (common_params->in_files.empty()) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Path cannot be empty");
        return nullptr;
    }
    auto path =  common_params->in_files[0];//also we may refactor for walk on in_files collection or read files from dirs
    if (path.empty()) {
        llama_dataset_set_error("GGUF path is null");
        return nullptr;
    }

    // Check if file exists
    FILE* file = fopen(path.c_str(), "rb");
    if (!file) {
        std::string msg = std::string("GGUF file not found ") + path;
        llama_dataset_set_error_with_code(DATASET_ERROR_FILE_NOT_FOUND, msg.c_str());
        return nullptr;
    }
    fclose(file);

    // Validate GGUF file format and detect corruption using comprehensive validation
    struct validation_result validation;
    if (!llama_dataset_validate_gguf_file(path.c_str(), &validation)) {
        llama_dataset_set_error_with_code(validation.error_code, validation.error_message);
        return nullptr;
    }

    // Create dataset structure
    struct llama_dataset* dataset = llama_dataset_alloc(DATASET_GGUF, common_params->dataset_streaming);
    if (!dataset) {
        // Error already set by dataset_alloc
        return nullptr;
    }

    // In streaming mode, we store the file path for later use
    if (common_params->dataset_streaming) {
        // Load GGUF context without tensor data
        struct gguf_init_params params;
        params.no_alloc = true;
        params.ctx = nullptr;

        dataset->ctx = gguf_init_from_file(path.c_str(), params);
        if (!dataset->ctx) {
            llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to load GGUF file");
            free(dataset);
            return nullptr;
        }

        dataset->format_data = strdup(path.c_str());
        if (!dataset->format_data) {
            llama_dataset_set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate path storage");
            gguf_free(dataset->ctx);
            free(dataset);
            return nullptr;
        }
    } else {
        // In non-streaming mode, we load all tensor data into memory
        struct gguf_init_params params;
        params.no_alloc = false;
        params.ctx = &dataset->ggml_ctx;

        dataset->ctx = gguf_init_from_file(path.c_str(), params);
        if (!dataset->ctx) {
            llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to load GGUF file");
            free(dataset);
            return nullptr;
        }

        // The GGML context is automatically created and tensors loaded by gguf_init_from_file
        if (!dataset->ggml_ctx) {
            llama_dataset_set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGML context");
            gguf_free(dataset->ctx);
            free(dataset);
            return nullptr;
        }

        // Validate that tensors were loaded correctly
        if (!llama_dataset_gguf_load_tensors(dataset->ctx, dataset->ggml_ctx)) {
            llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to validate GGUF tensors");
            ggml_free(dataset->ggml_ctx);
            gguf_free(dataset->ctx);
            free(dataset);
            return nullptr;
        }
    }

    // Cache tensor pointers for fast access
    if (!llama_dataset_cache_tensors(dataset)) {
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
void* llama_dataset_gguf_get_tensor_data_streaming(const struct llama_dataset* dataset, uint64_t index) {
    if (!dataset || !dataset->ctx || !dataset->streaming || !dataset->format_data) {
        llama_dataset_set_error("Invalid dataset for streaming data access");
        return nullptr;
    }

    // Check streaming cache first
    if (dataset->streaming_cache) {
        llama_dataset_streaming_cache* cache = dataset->streaming_cache;
        void* cached_data = cache->get(index);
        if (cached_data) {
            LLAMA_LOG_DEBUG("Retrieved sequence %zu from streaming cache\n", index);
            return cached_data;
        }
    }

    // Get tensor info from GGUF context
    const char* name = gguf_get_tensor_name(dataset->ctx, index);
    if (!name) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Invalid tensor index");
        return nullptr;
    }

    // Get tensor metadata
    int64_t tensor_id = gguf_find_tensor(dataset->ctx, name);
    if (tensor_id < 0) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Tensor not found in GGUF context");
        return nullptr;
    }

    // Get tensor size and offset
    size_t tensor_size = gguf_get_tensor_size(dataset->ctx, tensor_id);
    size_t tensor_offset = gguf_get_tensor_offset(dataset->ctx, tensor_id);
    size_t data_offset = gguf_get_data_offset(dataset->ctx);
    size_t file_offset = data_offset + tensor_offset;

    if (tensor_size == 0) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Invalid tensor size");
        return nullptr;
    }

    // Allocate memory for tensor data
    void* data = malloc(tensor_size);
    if (!data) {
        llama_dataset_set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate tensor data");
        return nullptr;
    }

    // Open file and seek to tensor data
    FILE* file = fopen(static_cast<const char *>(dataset->format_data), "rb");
    if (!file) {
        llama_dataset_set_error_with_code(DATASET_ERROR_FILE_NOT_FOUND, "Failed to open GGUF file for streaming");
        free(data);
        return nullptr;
    }

    // Seek to tensor data
    if (fseek(file, static_cast<long>(file_offset), SEEK_SET) != 0) {
        llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to seek to tensor data");
        fclose(file);
        free(data);
        return nullptr;
    }

    // Read tensor data
    if (fread(data, 1, tensor_size, file) != tensor_size) {
        llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to read tensor data");
        fclose(file);
        free(data);
        return nullptr;
    }

    fclose(file);

    // Add to streaming cache (cache takes ownership of the data)
    if (dataset->streaming_cache) {
        llama_dataset_streaming_cache* cache = dataset->streaming_cache;

        // Make a copy for the cache since the cache will manage the memory
        void* cache_data = malloc(tensor_size);
        if (cache_data) {
            memcpy(cache_data, data, tensor_size);
            cache->put(index, cache_data, tensor_size);
            LLAMA_LOG_DEBUG("Added sequence %zu to streaming cache (size: %zu bytes)\n", index, tensor_size);
        }
    }

    return data;
}
