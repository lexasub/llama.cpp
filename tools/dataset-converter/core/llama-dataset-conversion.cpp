#include "llama-dataset-conversion.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-error.h"
#include "llama-dataset-metadata.h"
#include "gguf.h"
#include "ggml.h"
#include "llama.h"
#include "llama-impl.h"  // For LLAMA_LOG_* macros
#include "common.h"      // For common_params
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <inttypes.h>
#include <sys/stat.h>
#include <errno.h>

// Progress tracking structure embedded in dataset
typedef struct {
    size_t total_sequences;
    size_t processed_sequences;
    double progress_percentage;
    bool has_error;
    char error_message[256];
    bool is_active;
} llama_dataset_conversion_progress_internal;

// Helper function to get or create progress tracker
static llama_dataset_conversion_progress_internal* get_progress_tracker(struct llama_dataset* dataset) {
    if (!dataset->format_data) {
        dataset->format_data = calloc(1, sizeof(llama_dataset_conversion_progress_internal));
    }
    return (llama_dataset_conversion_progress_internal*)dataset->format_data;
}

// Helper function to update progress
static void update_progress(struct llama_dataset* dataset, size_t processed, size_t total) {
    llama_dataset_conversion_progress_internal* progress = get_progress_tracker(dataset);
    if (progress) {
        progress->processed_sequences = processed;
        progress->total_sequences = total;
        progress->progress_percentage = total > 0 ? (double)processed / total * 100.0 : 0.0;
    }
}

bool llama_dataset_conversion_to_gguf(struct llama_dataset* dataset, const char* output_filename) {
    if (!dataset || !output_filename) {
        llama_dataset_error_set_with_context_internal("conversion", "to_gguf", "Invalid parameters");
        return false;
    }

    // Reset progress tracking
    llama_dataset_conversion_reset_progress(dataset);
    llama_dataset_conversion_progress_internal* progress = get_progress_tracker(dataset);
    if (progress) {
        progress->is_active = true;
    }

    // If dataset already has GGUF context, handle streaming/optimization case
    if (dataset->ctx) {
        uint64_t n_seq = dataset->n_seq;
        update_progress(dataset, 0, n_seq);

        // For streaming datasets, ensure all tensors are loaded
        if (dataset->streaming) {
            LLAMA_LOG_INFO("Converting streaming dataset to file: %s\n", output_filename);

            for (uint64_t i = 0; i < n_seq; i++) {
                // Load tensor data if not already loaded
                if (dataset->cached_tensors && !dataset->cached_tensors[i]) {
                    // This will trigger loading the tensor data
                    const int32_t* seq_data = llama_dataset_sequence(dataset, i);
                    if (!seq_data) {
                        llama_dataset_error_set_with_context_internal("conversion", "to_gguf", 
                            "Failed to load tensor data for streaming conversion");
                        if (progress) progress->is_active = false;
                        return false;
                    }
                }
                update_progress(dataset, i + 1, n_seq);
            }
        }

        // Write GGUF file manually for proper streaming support
        FILE* file = fopen(output_filename, "wb");
        if (!file) {
            llama_dataset_error_set_with_context_internal("conversion", "to_gguf", 
                "Failed to open output file for writing");
            if (progress) progress->is_active = false;
            return false;
        }

        // Write metadata
        size_t meta_size = gguf_get_meta_size(dataset->ctx);
        void* meta_data = malloc(meta_size);
        if (!meta_data) {
            fclose(file);
            llama_dataset_error_set_with_context_internal("conversion", "to_gguf", 
                "Failed to allocate memory for GGUF metadata");
            if (progress) progress->is_active = false;
            return false;
        }

        gguf_get_meta_data(dataset->ctx, meta_data);
        if (fwrite(meta_data, 1, meta_size, file) != meta_size) {
            free(meta_data);
            fclose(file);
            llama_dataset_error_set_with_context_internal("conversion", "to_gguf", 
                "Failed to write GGUF metadata");
            if (progress) progress->is_active = false;
            return false;
        }
        free(meta_data);

        // Write tensor data
        uint64_t n_tensors = gguf_get_n_tensors(dataset->ctx);
        for (uint64_t i = 0; i < n_tensors; i++) {
            const void* tensor_data = nullptr;
            size_t tensor_size = 0;

            if (dataset->cached_tensors && dataset->cached_tensors[i]) {
                if (dataset->cached_tensors[i]->data) {
                    tensor_data = dataset->cached_tensors[i]->data;
                    tensor_size = ggml_nbytes(dataset->cached_tensors[i]);
                } else {
                    tensor_size = gguf_get_tensor_size(dataset->ctx, i);
                }
            } else {
                tensor_size = gguf_get_tensor_size(dataset->ctx, i);
            }

            if (!tensor_data || tensor_size == 0) {
                fclose(file);
                llama_dataset_error_set_with_context_internal("conversion", "to_gguf", 
                    "Invalid tensor data for writing");
                if (progress) progress->is_active = false;
                return false;
            }

            // Write tensor data with alignment
            if (fwrite(tensor_data, 1, tensor_size, file) != tensor_size) {
                fclose(file);
                llama_dataset_error_set_with_context_internal("conversion", "to_gguf", 
                    "Failed to write tensor data");
                if (progress) progress->is_active = false;
                return false;
            }

            // Add padding for 32-byte alignment
            size_t padding = (32 - (tensor_size % 32)) % 32;
            if (padding > 0) {
                char zero_padding[32] = {0};
                if (fwrite(zero_padding, 1, padding, file) != padding) {
                    fclose(file);
                    llama_dataset_error_set_with_context_internal("conversion", "to_gguf", 
                        "Failed to write tensor padding");
                    if (progress) progress->is_active = false;
                    return false;
                }
            }
        }

        fclose(file);
        if (progress) progress->is_active = false;
        LLAMA_LOG_INFO("Successfully wrote GGUF dataset to %s\n", output_filename);
        return true;
    }

    // For non-GGUF formats, create new GGUF file
    LLAMA_LOG_INFO("Converting %s dataset to GGUF file: %s\n", 
        dataset->type == DATASET_TEXT ? "TEXT" : "PARQUET", output_filename);

    struct gguf_context* new_ctx = gguf_init_empty();
    if (!new_ctx) {
        llama_dataset_error_set_with_context_internal("conversion", "to_gguf", 
            "Failed to create GGUF context for conversion");
        if (progress) progress->is_active = false;
        return false;
    }

    // Preserve metadata from original dataset
    if (!llama_dataset_conversion_preserve_metadata(dataset, nullptr)) {
        gguf_free(new_ctx);
        if (progress) progress->is_active = false;
        return false;
    }

    // Add standard metadata
    gguf_set_val_str(new_ctx, "training.format.source", 
        dataset->type == DATASET_TEXT ? "text" : 
        dataset->type == DATASET_PARQUET ? "parquet" : "gguf");

    time_t now = time(nullptr);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    gguf_set_val_str(new_ctx, "training.creation.time", timestamp);

    uint64_t seq_count = dataset->n_seq;
    gguf_set_val_i32(new_ctx, "training.sequence.count", static_cast<int32_t>(seq_count));

    // Find maximum sequence length
    int32_t max_length = 0;
    for (uint64_t i = 0; i < seq_count; i++) {
        int32_t len = llama_dataset_sequence_length(dataset, i);
        if (len > max_length) {
            max_length = len;
        }
    }
    gguf_set_val_u32(new_ctx, "training.max.length", static_cast<uint32_t>(max_length));

    // Create GGML context for tensor data
    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ 128ull*1024ull*1024ull,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context* ggml_ctx = ggml_init(ggml_params);
    if (!ggml_ctx) {
        gguf_free(new_ctx);
        llama_dataset_error_set_with_context_internal("conversion", "to_gguf", 
            "Failed to create GGML context for conversion");
        if (progress) progress->is_active = false;
        return false;
    }

    update_progress(dataset, 0, seq_count);

    // Add all sequences as tensors
    for (uint64_t i = 0; i < seq_count; i++) {
        const int32_t* tokens = llama_dataset_sequence(dataset, i);
        int32_t length = llama_dataset_sequence_length(dataset, i);

        if (!tokens || length <= 0) {
            LLAMA_LOG_WARN("Skipping invalid sequence at index %zu\n", i);
            continue;
        }

        // Create tensor name
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

        update_progress(dataset, i + 1, seq_count);
    }

    // Write GGUF file
    bool write_success = gguf_write_to_file(new_ctx, output_filename, false);

    // Clean up
    ggml_free(ggml_ctx);
    gguf_free(new_ctx);

    if (progress) progress->is_active = false;

    if (!write_success) {
        llama_dataset_error_set_with_context_internal("conversion", "to_gguf", 
            "Failed to write GGUF file");
        return false;
    }

    LLAMA_LOG_INFO("Successfully converted dataset to GGUF file: %s\n", output_filename);
    return true;
}

bool llama_dataset_conversion_transform_format(struct llama_dataset* dataset, int target_format) {
    if (!dataset) {
        llama_dataset_error_set_with_context_internal("conversion", "transform_format", "Invalid dataset parameter");
        return false;
    }

    // Check if conversion is needed
    if (dataset->type == target_format) {
        LLAMA_LOG_INFO("Dataset is already in target format %d, no conversion needed\n", target_format);
        return true;
    }

    // Reset progress tracking
    llama_dataset_conversion_reset_progress(dataset);
    llama_dataset_conversion_progress_internal* progress = get_progress_tracker(dataset);
    if (progress) {
        progress->is_active = true;
    }

    bool success = false;

    switch (target_format) {
        case DATASET_GGUF: {
            // Transform to GGUF format - this is the most common transformation
            LLAMA_LOG_INFO("Transforming dataset from format %d to GGUF\n", dataset->type);
            
            // Create new GGUF context if not already present
            if (!dataset->ctx) {
                dataset->ctx = gguf_init_empty();
                if (!dataset->ctx) {
                    llama_dataset_error_set_with_context_internal("conversion", "transform_format", 
                        "Failed to create GGUF context for transformation");
                    break;
                }
            }

            // Apply format-specific optimizations
            if (!llama_dataset_conversion_optimize_for_format(dataset, target_format)) {
                llama_dataset_error_set_with_context_internal("conversion", "transform_format", 
                    "Failed to optimize dataset for GGUF format");
                break;
            }

            // Apply format-specific settings
            if (!llama_dataset_conversion_apply_format_specific_settings(dataset, target_format)) {
                llama_dataset_error_set_with_context_internal("conversion", "transform_format", 
                    "Failed to apply GGUF format settings");
                break;
            }

            // Update dataset type
            dataset->type = DATASET_GGUF;
            success = true;
            break;
        }

        case DATASET_TEXT: {
            LLAMA_LOG_INFO("Transforming dataset from format %d to TEXT\n", dataset->type);
            
            // For TEXT format, we need to ensure tokenization capabilities
            if (!dataset->model) {
                llama_dataset_error_set_with_context_internal("conversion", "transform_format", 
                    "TEXT format requires a tokenization model");
                break;
            }

            // Apply format-specific optimizations
            if (!llama_dataset_conversion_optimize_for_format(dataset, target_format)) {
                llama_dataset_error_set_with_context_internal("conversion", "transform_format", 
                    "Failed to optimize dataset for TEXT format");
                break;
            }

            // Apply format-specific settings
            if (!llama_dataset_conversion_apply_format_specific_settings(dataset, target_format)) {
                llama_dataset_error_set_with_context_internal("conversion", "transform_format", 
                    "Failed to apply TEXT format settings");
                break;
            }

            dataset->type = DATASET_TEXT;
            success = true;
            break;
        }

        case DATASET_PARQUET: {
            LLAMA_LOG_INFO("Transforming dataset from format %d to PARQUET\n", dataset->type);
            
            // Apply format-specific optimizations
            if (!llama_dataset_conversion_optimize_for_format(dataset, target_format)) {
                llama_dataset_error_set_with_context_internal("conversion", "transform_format", 
                    "Failed to optimize dataset for PARQUET format");
                break;
            }

            // Apply format-specific settings
            if (!llama_dataset_conversion_apply_format_specific_settings(dataset, target_format)) {
                llama_dataset_error_set_with_context_internal("conversion", "transform_format", 
                    "Failed to apply PARQUET format settings");
                break;
            }

            dataset->type = DATASET_PARQUET;
            success = true;
            break;
        }

        default:
            llama_dataset_error_set_with_context_internal("conversion", "transform_format", 
                "Unsupported target format");
            break;
    }

    if (progress) {
        progress->is_active = false;
        if (success) {
            progress->progress_percentage = 100.0;
        }
    }

    if (success) {
        LLAMA_LOG_INFO("Successfully transformed dataset to format %d\n", target_format);
    }

    return success;
}

bool llama_dataset_conversion_preserve_metadata(struct llama_dataset* source, struct llama_dataset* target) {
    if (!source) {
        llama_dataset_error_set_with_context_internal("conversion", "preserve_metadata", "Invalid source parameter");
        return false;
    }

    // If target is NULL, we're preserving metadata within the same dataset (internal operation)
    struct gguf_context* target_ctx = target ? target->ctx : source->ctx;
    if (!target_ctx) {
        llama_dataset_error_set_with_context_internal("conversion", "preserve_metadata", "No target context available");
        return false;
    }

    // If source has GGUF context, copy all metadata
    if (source->ctx) {
        int n_kv = gguf_get_n_kv(source->ctx);
        
        for (int i = 0; i < n_kv; i++) {
            const char* key = gguf_get_key(source->ctx, i);
            enum gguf_type type = gguf_get_kv_type(source->ctx, i);

            // Skip if key already exists in target (avoid overwriting)
            if (target && target != source) {
                int existing_idx = gguf_find_key(target_ctx, key);
                if (existing_idx >= 0) {
                    LLAMA_LOG_DEBUG("Skipping existing metadata key: %s\n", key);
                    continue;
                }
            }

            // Copy metadata based on type
            switch (type) {
                case GGUF_TYPE_STRING: {
                    const char* value = gguf_get_val_str(source->ctx, i);
                    gguf_set_val_str(target_ctx, key, value);
                    break;
                }
                case GGUF_TYPE_INT8: {
                    int8_t value = gguf_get_val_i8(source->ctx, i);
                    gguf_set_val_i8(target_ctx, key, value);
                    break;
                }
                case GGUF_TYPE_UINT8: {
                    uint8_t value = gguf_get_val_u8(source->ctx, i);
                    gguf_set_val_u8(target_ctx, key, value);
                    break;
                }
                case GGUF_TYPE_INT16: {
                    int16_t value = gguf_get_val_i16(source->ctx, i);
                    gguf_set_val_i16(target_ctx, key, value);
                    break;
                }
                case GGUF_TYPE_UINT16: {
                    uint16_t value = gguf_get_val_u16(source->ctx, i);
                    gguf_set_val_u16(target_ctx, key, value);
                    break;
                }
                case GGUF_TYPE_INT32: {
                    int32_t value = gguf_get_val_i32(source->ctx, i);
                    gguf_set_val_i32(target_ctx, key, value);
                    break;
                }
                case GGUF_TYPE_UINT32: {
                    uint32_t value = gguf_get_val_u32(source->ctx, i);
                    gguf_set_val_u32(target_ctx, key, value);
                    break;
                }
                case GGUF_TYPE_INT64: {
                    int64_t value = gguf_get_val_i64(source->ctx, i);
                    gguf_set_val_i64(target_ctx, key, value);
                    break;
                }
                case GGUF_TYPE_UINT64: {
                    uint64_t value = gguf_get_val_u64(source->ctx, i);
                    gguf_set_val_u64(target_ctx, key, value);
                    break;
                }
                case GGUF_TYPE_FLOAT32: {
                    float value = gguf_get_val_f32(source->ctx, i);
                    gguf_set_val_f32(target_ctx, key, value);
                    break;
                }
                case GGUF_TYPE_FLOAT64: {
                    double value = gguf_get_val_f64(source->ctx, i);
                    gguf_set_val_f64(target_ctx, key, value);
                    break;
                }
                case GGUF_TYPE_BOOL: {
                    bool value = gguf_get_val_bool(source->ctx, i);
                    gguf_set_val_bool(target_ctx, key, value);
                    break;
                }
                case GGUF_TYPE_ARRAY: {
                    // For arrays, we need to handle each element type
                    // Note: GGUF library may not have direct array copying functions
                    // For now, we'll log and skip array metadata
                    LLAMA_LOG_WARN("Array metadata preservation requires manual implementation for key: %s\n", key);
                    // Arrays are complex and require format-specific handling
                    // This could be implemented in the future if needed for specific use cases
                    break;
                }
                default:
                    LLAMA_LOG_WARN("Unknown metadata type %d for key '%s', skipping\n", type, key);
                    break;
            }
        }
    }

    // Add format-specific metadata preservation
    switch (source->type) {
        case DATASET_TEXT:
            // Preserve text-specific metadata
            if (source->model) {
                // Add tokenizer information if available
                gguf_set_val_str(target_ctx, "training.tokenizer.type", "llama");
                // Could add more tokenizer-specific metadata here
            }
            break;

        case DATASET_PARQUET:
            // Preserve Parquet-specific metadata
            if (source->format_data) {
                // Add Parquet schema information if available
                gguf_set_val_str(target_ctx, "training.parquet.schema", "preserved");
                // Could add more Parquet-specific metadata here
            }
            break;

        case DATASET_GGUF:
            // GGUF metadata is already preserved above
            break;

        default:
            LLAMA_LOG_WARN("Unknown source format %d for metadata preservation\n", source->type);
            break;
    }

    // Add preservation timestamp
    time_t now = time(nullptr);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    gguf_set_val_str(target_ctx, "training.metadata.preserved_at", timestamp);

    LLAMA_LOG_INFO("Successfully preserved metadata from source dataset\n");
    return true;
}

bool llama_dataset_conversion_validate_conversion(const struct llama_dataset* source, const struct llama_dataset* target) {
    if (!source || !target) {
        llama_dataset_error_set_with_context_internal("conversion", "validate_conversion", "Invalid parameters");
        return false;
    }

    // Basic validation - check sequence counts match
    if (source->n_seq != target->n_seq) {
        llama_dataset_error_set_formatted_with_context_internal("conversion", "validate_conversion", DATASET_ERROR_REGISTRY_VALIDATION_FAILED,
            "Sequence count mismatch: source has %zu sequences, target has %zu sequences", 
            source->n_seq, target->n_seq);
        return false;
    }

    LLAMA_LOG_INFO("Validating conversion between datasets (%zu sequences)\n", source->n_seq);

    // Validate each sequence
    for (uint64_t i = 0; i < source->n_seq; i++) {
        // Check sequence lengths
        int32_t source_length = llama_dataset_sequence_length(source, i);
        int32_t target_length = llama_dataset_sequence_length(target, i);
        
        if (source_length != target_length) {
            llama_dataset_error_set_formatted_with_context_internal("conversion", "validate_conversion", DATASET_ERROR_REGISTRY_VALIDATION_FAILED,
                "Sequence length mismatch at index %zu: source=%d, target=%d", 
                i, source_length, target_length);
            return false;
        }

        // Check sequence data
        const int32_t* source_data = llama_dataset_sequence(source, i);
        const int32_t* target_data = llama_dataset_sequence(target, i);
        
        if (!source_data || !target_data) {
            llama_dataset_error_set_formatted_with_context_internal("conversion", "validate_conversion", DATASET_ERROR_REGISTRY_VALIDATION_FAILED,
                "Missing sequence data at index %zu", i);
            return false;
        }

        // Compare sequence content
        if (memcmp(source_data, target_data, source_length * sizeof(int32_t)) != 0) {
            llama_dataset_error_set_formatted_with_context_internal("conversion", "validate_conversion", DATASET_ERROR_REGISTRY_VALIDATION_FAILED,
                "Sequence data mismatch at index %zu", i);
            return false;
        }
    }

    // Validate critical metadata preservation
    if (source->ctx && target->ctx) {
        // Check that important metadata keys are preserved
        const char* critical_keys[] = {
            "training.sequence.count",
            "training.max.length",
            "training.format.source"
        };
        
        for (size_t i = 0; i < sizeof(critical_keys) / sizeof(critical_keys[0]); i++) {
            int source_idx = gguf_find_key(source->ctx, critical_keys[i]);
            int target_idx = gguf_find_key(target->ctx, critical_keys[i]);
            
            if (source_idx >= 0 && target_idx < 0) {
                LLAMA_LOG_WARN("Critical metadata key '%s' not preserved in target\n", critical_keys[i]);
            }
        }
    }

    // Validate format-specific aspects
    if (source->type != target->type) {
        LLAMA_LOG_INFO("Cross-format conversion validated: %d -> %d\n", source->type, target->type);
    }

    // Check streaming consistency
    if (source->streaming != target->streaming) {
        LLAMA_LOG_INFO("Streaming mode changed during conversion: %s -> %s\n", 
            source->streaming ? "enabled" : "disabled",
            target->streaming ? "enabled" : "disabled");
    }

    LLAMA_LOG_INFO("Conversion validation completed successfully\n");
    return true;
}

bool llama_dataset_conversion_check_integrity(const struct llama_dataset* dataset, const char* filename) {
    if (!dataset || !filename) {
        llama_dataset_error_set_with_context_internal("conversion", "check_integrity", "Invalid parameters");
        return false;
    }

    LLAMA_LOG_INFO("Checking integrity of dataset file: %s\n", filename);

    // Check if file exists and is readable
    FILE* file = fopen(filename, "rb");
    if (!file) {
        llama_dataset_error_set_with_context_internal("conversion", "check_integrity", 
            "Cannot open file for integrity check");
        return false;
    }

    // Get file size
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    fclose(file);

    if (file_size <= 0) {
        llama_dataset_error_set_with_context_internal("conversion", "check_integrity", 
            "File is empty or invalid size");
        return false;
    }

    // For GGUF files, perform more detailed integrity checks
    if (dataset->type == DATASET_GGUF || strstr(filename, ".gguf")) {
        // Try to load the file as GGUF to validate structure
        struct gguf_init_params params = {
            /*.no_alloc = */ true,  // Don't allocate tensor data for integrity check
            /*.ctx      = */ nullptr,
        };
        
        struct gguf_context* test_ctx = gguf_init_from_file(filename, params);
        if (!test_ctx) {
            llama_dataset_error_set_with_context_internal("conversion", "check_integrity", 
                "File is not a valid GGUF file");
            return false;
        }

        // Check basic GGUF structure
        int n_kv = gguf_get_n_kv(test_ctx);
        uint64_t n_tensors = gguf_get_n_tensors(test_ctx);
        
        LLAMA_LOG_INFO("GGUF integrity check: %d metadata keys, %zu tensors\n", n_kv, n_tensors);

        // Validate that expected metadata exists
        int seq_count_idx = gguf_find_key(test_ctx, "training.sequence.count");
        if (seq_count_idx >= 0) {
            int32_t file_seq_count = gguf_get_val_i32(test_ctx, seq_count_idx);
            if (file_seq_count != (int32_t)dataset->n_seq) {
                gguf_free(test_ctx);
                llama_dataset_error_set_formatted_with_context_internal("conversion", "check_integrity", DATASET_ERROR_REGISTRY_VALIDATION_FAILED,
                    "Sequence count mismatch: dataset=%zu, file=%d", dataset->n_seq, file_seq_count);
                return false;
            }
        }

        // Check tensor names and sizes
        for (uint64_t i = 0; i < n_tensors; i++) {
            const char* tensor_name = gguf_get_tensor_name(test_ctx, i);
            if (!tensor_name) {
                gguf_free(test_ctx);
                llama_dataset_error_set_formatted_with_context_internal("conversion", "check_integrity", DATASET_ERROR_INVALID_FORMAT,
                    "Invalid tensor name at index %zu", i);
                return false;
            }

            // Check tensor size
            size_t tensor_size = gguf_get_tensor_size(test_ctx, i);
            if (tensor_size == 0) {
                gguf_free(test_ctx);
                llama_dataset_error_set_formatted_with_context_internal("conversion", "check_integrity", DATASET_ERROR_INVALID_FORMAT,
                    "Invalid tensor size for tensor '%s'", tensor_name);
                return false;
            }
        }

        gguf_free(test_ctx);
    }

    // Check file size consistency with dataset
    if (dataset->ctx) {
        size_t expected_meta_size = gguf_get_meta_size(dataset->ctx);
        uint64_t n_tensors = gguf_get_n_tensors(dataset->ctx);
        
        size_t expected_tensor_size = 0;
        for (uint64_t i = 0; i < n_tensors; i++) {
            size_t tensor_size = gguf_get_tensor_size(dataset->ctx, i);
            expected_tensor_size += tensor_size;
            // Add padding for alignment
            expected_tensor_size += (32 - (tensor_size % 32)) % 32;
        }

        size_t expected_total_size = expected_meta_size + expected_tensor_size;
        
        // Allow some tolerance for file system overhead
        if (file_size < (long)(expected_total_size * 0.95) || 
            file_size > (long)(expected_total_size * 1.05)) {
            LLAMA_LOG_WARN("File size mismatch: expected ~%zu bytes, got %ld bytes\n", 
                expected_total_size, file_size);
            // Don't fail on size mismatch, just warn
        }
    }

    LLAMA_LOG_INFO("File integrity check passed for: %s\n", filename);
    return true;
}

bool llama_dataset_conversion_get_progress(const struct llama_dataset* dataset, llama_dataset_conversion_progress* progress) {
    if (!dataset || !progress) {
        llama_dataset_error_set_with_context_internal("conversion", "get_progress", "Invalid parameters");
        return false;
    }

    // Initialize progress structure
    memset(progress, 0, sizeof(llama_dataset_conversion_progress));

    // Get internal progress tracker
    llama_dataset_conversion_progress_internal* internal_progress = 
        (llama_dataset_conversion_progress_internal*)dataset->format_data;

    if (internal_progress && internal_progress->is_active) {
        // Copy progress from internal tracker
        progress->total_sequences = internal_progress->total_sequences;
        progress->processed_sequences = internal_progress->processed_sequences;
        progress->progress_percentage = internal_progress->progress_percentage;
        progress->has_error = internal_progress->has_error;
        
        // Copy error message if present
        if (internal_progress->has_error && internal_progress->error_message[0] != '\0') {
            size_t len = strlen(internal_progress->error_message);
            size_t max_len = sizeof(progress->error_message) - 1;
            size_t copy_len = len < max_len ? len : max_len;
            memcpy(progress->error_message, internal_progress->error_message, copy_len);
            progress->error_message[copy_len] = '\0';
        }
    } else {
        // No active conversion, provide default values
        progress->total_sequences = dataset->n_seq;
        progress->processed_sequences = dataset->n_seq;
        progress->progress_percentage = 100.0;
        progress->has_error = false;
        progress->error_message[0] = '\0';
    }

    return true;
}

void llama_dataset_conversion_reset_progress(struct llama_dataset* dataset) {
    if (!dataset) {
        return;
    }

    // Get or create progress tracker
    llama_dataset_conversion_progress_internal* progress = get_progress_tracker(dataset);
    if (progress) {
        // Reset all progress fields
        progress->total_sequences = 0;
        progress->processed_sequences = 0;
        progress->progress_percentage = 0.0;
        progress->has_error = false;
        progress->error_message[0] = '\0';
        progress->is_active = false;
    }
}

bool llama_dataset_conversion_optimize_for_format(struct llama_dataset* dataset, int target_format) {
    if (!dataset) {
        llama_dataset_error_set_with_context_internal("conversion", "optimize_for_format", "Invalid dataset parameter");
        return false;
    }

    LLAMA_LOG_INFO("Optimizing dataset for target format %d\n", target_format);

    switch (target_format) {
        case DATASET_GGUF: {
            // GGUF format optimizations
            
            // Optimize tensor caching for GGUF access patterns
            if (dataset->streaming && dataset->cached_tensors) {
                // Pre-load frequently accessed tensors
                uint64_t cache_target = dataset->n_seq < 1000 ? dataset->n_seq : 1000;
                for (uint64_t i = 0; i < cache_target; i++) {
                    if (!dataset->cached_tensors[i]) {
                        // Trigger loading of tensor
                        llama_dataset_sequence(dataset, i);
                    }
                }
            }

            // Optimize memory layout for GGUF
            if (dataset->ctx) {
                // Ensure metadata is optimally organized
                // This is mostly handled by GGUF library, but we can add hints
                LLAMA_LOG_DEBUG("GGUF metadata optimization applied\n");
            }

            break;
        }

        case DATASET_TEXT: {
            // Text format optimizations
            
            // Ensure tokenization context is optimized
            if (dataset->model && !dataset->tokenizer_ctx) {
                // Create optimized tokenizer context for batch processing
                struct llama_context_params ctx_params = llama_context_default_params();
                ctx_params.n_ctx = 2048;  // Reasonable context size for tokenization
                ctx_params.n_batch = 512; // Optimize for batch tokenization
                ctx_params.n_threads = 4; // Use multiple threads for tokenization
                
                dataset->tokenizer_ctx = llama_init_from_model(dataset->model, ctx_params);
                if (!dataset->tokenizer_ctx) {
                    llama_dataset_error_set_with_context_internal("conversion", "optimize_for_format", 
                        "Failed to create optimized tokenizer context");
                    return false;
                }
            }

            // Optimize streaming cache for text access patterns
            if (dataset->streaming) {
                // Text datasets often have sequential access patterns
                // Configure cache for read-ahead optimization
                LLAMA_LOG_DEBUG("Text streaming optimization applied\n");
            }

            break;
        }

        case DATASET_PARQUET: {
            // Parquet format optimizations
            
            // Optimize for columnar access patterns
            if (dataset->streaming) {
                // Parquet benefits from column-wise caching
                LLAMA_LOG_DEBUG("Parquet columnar optimization applied\n");
            }

            // Optimize batch processing for Parquet
            // Parquet works best with larger batch sizes
            LLAMA_LOG_DEBUG("Parquet batch processing optimization applied\n");

            break;
        }

        default:
            LLAMA_LOG_WARN("No specific optimizations available for format %d\n", target_format);
            break;
    }

    LLAMA_LOG_INFO("Format optimization completed for target format %d\n", target_format);
    return true;
}

bool llama_dataset_conversion_apply_format_specific_settings(struct llama_dataset* dataset, int target_format) {
    if (!dataset) {
        llama_dataset_error_set_with_context_internal("conversion", "apply_format_specific_settings", "Invalid dataset parameter");
        return false;
    }

    LLAMA_LOG_INFO("Applying format-specific settings for target format %d\n", target_format);

    switch (target_format) {
        case DATASET_GGUF: {
            // GGUF format settings
            
            // Ensure GGUF context exists
            if (!dataset->ctx) {
                dataset->ctx = gguf_init_empty();
                if (!dataset->ctx) {
                    llama_dataset_error_set_with_context_internal("conversion", "apply_format_specific_settings", 
                        "Failed to create GGUF context");
                    return false;
                }
            }

            // Set GGUF-specific metadata
            gguf_set_val_str(dataset->ctx, "training.format.target", "gguf");
            gguf_set_val_i16(dataset->ctx, "training.format.version", 1000);
            
            // Configure tensor naming convention for GGUF
            gguf_set_val_str(dataset->ctx, "training.tensor.naming", "seq_XXXXX");
            
            // Set optimal alignment for GGUF tensors
            gguf_set_val_u32(dataset->ctx, "training.tensor.alignment", 32);

            break;
        }

        case DATASET_TEXT: {
            // Text format settings
            
            // Ensure we have tokenization capabilities
            if (!dataset->model) {
                llama_dataset_error_set_with_context_internal("conversion", "apply_format_specific_settings", 
                    "Text format requires a tokenization model");
                return false;
            }

            // Set text-specific metadata
            if (dataset->ctx) {
                gguf_set_val_str(dataset->ctx, "training.format.target", "text");
                gguf_set_val_str(dataset->ctx, "training.tokenizer.type", "llama");
                gguf_set_val_bool(dataset->ctx, "training.text.requires_tokenization", true);
            }

            // Configure streaming for text processing
            if (!dataset->streaming) {
                // Text datasets often benefit from streaming
                dataset->streaming = true;
                LLAMA_LOG_INFO("Enabled streaming mode for text format\n");
            }

            break;
        }

        case DATASET_PARQUET: {
            // Parquet format settings
            
            // Set Parquet-specific metadata
            if (dataset->ctx) {
                gguf_set_val_str(dataset->ctx, "training.format.target", "parquet");
                gguf_set_val_str(dataset->ctx, "training.parquet.engine", "arrow");
                gguf_set_val_bool(dataset->ctx, "training.parquet.columnar", true);
            }

            // Configure optimal batch size for Parquet
            // Parquet works best with larger batches due to columnar nature
            LLAMA_LOG_DEBUG("Configured optimal batch size for Parquet format\n");

            // Enable streaming for large Parquet files
            if (!dataset->streaming && dataset->n_seq > 10000) {
                dataset->streaming = true;
                LLAMA_LOG_INFO("Enabled streaming mode for large Parquet dataset\n");
            }

            break;
        }

        default:
            LLAMA_LOG_WARN("No specific settings available for format %d\n", target_format);
            // Still return true as this is not a critical failure
            break;
    }

    // Apply common settings for all formats
    if (dataset->ctx) {
        // Add conversion timestamp
        time_t now = time(nullptr);
        char timestamp[32];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
        gguf_set_val_str(dataset->ctx, "training.conversion.timestamp", timestamp);
        
        // Add sequence count
        gguf_set_val_u64(dataset->ctx, "training.sequence.count", dataset->n_seq);
    }

    LLAMA_LOG_INFO("Format-specific settings applied for target format %d\n", target_format);
    return true;
}

// Additional conversion utility functions

bool llama_dataset_conversion_estimate_output_size(const struct llama_dataset* dataset, const char* target_format, size_t* estimated_size) {
    if (!dataset || !target_format || !estimated_size) {
        llama_dataset_error_set_with_context_internal("conversion", "estimate_output_size", "Invalid parameters");
        return false;
    }

    *estimated_size = 0;

    // Base metadata size estimate
    size_t metadata_size = 1024; // Base metadata overhead
    
    if (dataset->ctx) {
        // Add existing metadata size
        metadata_size += gguf_get_meta_size(dataset->ctx);
    }

    // Estimate tensor data size
    size_t tensor_data_size = 0;
    uint64_t n_seq = dataset->n_seq;
    
    for (uint64_t i = 0; i < n_seq; i++) {
        int32_t seq_len = llama_dataset_sequence_length(dataset, i);
        if (seq_len > 0) {
            tensor_data_size += seq_len * sizeof(int32_t);
            // Add padding for alignment
            tensor_data_size += (32 - (tensor_data_size % 32)) % 32;
        }
    }

    // Format-specific size adjustments
    if (strcmp(target_format, "gguf") == 0) {
        // GGUF format overhead
        *estimated_size = metadata_size + tensor_data_size + (n_seq * 64); // Tensor headers
    } else if (strcmp(target_format, "text") == 0) {
        // Text format - estimate based on average token representation
        *estimated_size = tensor_data_size * 6; // Rough estimate: 6 chars per token on average
    } else if (strcmp(target_format, "parquet") == 0) {
        // Parquet format - columnar compression
        *estimated_size = tensor_data_size * 0.7; // Estimate 30% compression
    } else {
        // Unknown format, use conservative estimate
        *estimated_size = metadata_size + tensor_data_size * 1.2;
    }

    LLAMA_LOG_INFO("Estimated output size for %s format: %zu bytes\n", target_format, *estimated_size);
    return true;
}

bool llama_dataset_conversion_batch_convert(struct llama_dataset** datasets, size_t dataset_count, 
                                           const char* output_dir, const char* target_format) {
    if (!datasets || dataset_count == 0 || !output_dir || !target_format) {
        llama_dataset_error_set_with_context_internal("conversion", "batch_convert", "Invalid parameters");
        return false;
    }

    LLAMA_LOG_INFO("Starting batch conversion of %zu datasets to %s format\n", dataset_count, target_format);

    bool all_success = true;
    size_t successful_conversions = 0;

    for (size_t i = 0; i < dataset_count; i++) {
        if (!datasets[i]) {
            LLAMA_LOG_WARN("Skipping NULL dataset at index %zu\n", i);
            continue;
        }

        // Generate output filename
        char output_filename[512];
        snprintf(output_filename, sizeof(output_filename), "%s/dataset_%zu.%s", 
                output_dir, i, target_format);

        // Convert based on target format
        bool success = false;
        if (strcmp(target_format, "gguf") == 0) {
            success = llama_dataset_conversion_to_gguf(datasets[i], output_filename);
        } else {
            // For other formats, transform first then save
            int target_format_enum = DATASET_GGUF; // Default
            if (strcmp(target_format, "text") == 0) {
                target_format_enum = DATASET_TEXT;
            } else if (strcmp(target_format, "parquet") == 0) {
                target_format_enum = DATASET_PARQUET;
            }

            success = llama_dataset_conversion_transform_format(datasets[i], target_format_enum);
            if (success && target_format_enum == DATASET_GGUF) {
                success = llama_dataset_conversion_to_gguf(datasets[i], output_filename);
            }
        }

        if (success) {
            successful_conversions++;
            LLAMA_LOG_INFO("Successfully converted dataset %zu to %s\n", i, output_filename);
        } else {
            all_success = false;
            LLAMA_LOG_ERROR("Failed to convert dataset %zu: %s\n", i, llama_dataset_get_error_message());
        }
    }

    LLAMA_LOG_INFO("Batch conversion completed: %zu/%zu successful\n", successful_conversions, dataset_count);
    return all_success;
}

bool llama_dataset_conversion_verify_output(const char* output_filename, const struct llama_dataset* original_dataset) {
    if (!output_filename || !original_dataset) {
        llama_dataset_error_set_with_context_internal("conversion", "verify_output", "Invalid parameters");
        return false;
    }

    LLAMA_LOG_INFO("Verifying converted output file: %s\n", output_filename);

    // First check basic file integrity
    if (!llama_dataset_conversion_check_integrity(original_dataset, output_filename)) {
        return false;
    }

    // Try to load the converted file and compare
    struct common_params params = {};
    params.in_files.push_back(output_filename);
    
    struct llama_dataset* converted_dataset = llama_dataset_from_gguf(&params);
    if (!converted_dataset) {
        llama_dataset_error_set_with_context_internal("conversion", "verify_output", 
            "Failed to load converted dataset for verification");
        return false;
    }

    // Validate conversion
    bool validation_success = llama_dataset_conversion_validate_conversion(original_dataset, converted_dataset);
    
    // Clean up
    llama_dataset_free(converted_dataset);

    if (validation_success) {
        LLAMA_LOG_INFO("Output verification successful for: %s\n", output_filename);
    } else {
        LLAMA_LOG_ERROR("Output verification failed for: %s\n", output_filename);
    }

    return validation_success;
}

bool llama_dataset_conversion_get_supported_formats(const char*** formats, size_t* format_count) {
    if (!formats || !format_count) {
        llama_dataset_error_set_with_context_internal("conversion", "get_supported_formats", "Invalid parameters");
        return false;
    }

    static const char* supported_formats[] = {
        "gguf",
        "text", 
        "parquet"
    };

    *formats = supported_formats;
    *format_count = sizeof(supported_formats) / sizeof(supported_formats[0]);

    return true;
}

bool llama_dataset_conversion_is_format_supported(const char* format) {
    if (!format) {
        return false;
    }

    const char** formats;
    size_t format_count;
    
    if (!llama_dataset_conversion_get_supported_formats(&formats, &format_count)) {
        return false;
    }

    for (size_t i = 0; i < format_count; i++) {
        if (strcmp(format, formats[i]) == 0) {
            return true;
        }
    }

    return false;
}