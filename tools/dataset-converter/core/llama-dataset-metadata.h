#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct llama_dataset;
struct gguf_context;

// Internal metadata access interface
const char* llama_dataset_metadata_get_str_internal(const struct llama_dataset* dataset, const char* key);
int64_t llama_dataset_metadata_get_int_internal(const struct llama_dataset* dataset, const char* key, int64_t default_value);
float llama_dataset_metadata_get_float_internal(const struct llama_dataset* dataset, const char* key, float default_value);

// Metadata validation and normalization with comprehensive error handling
bool llama_dataset_metadata_validate_key_internal(const char* key);
const char* llama_dataset_metadata_normalize_key_internal(const char* key);
bool llama_dataset_metadata_validate_key_with_error_internal(const char* key);

// Enhanced type conversion utilities with comprehensive error handling
bool llama_dataset_metadata_convert_to_int_internal(const char* str_value, int64_t* result);
bool llama_dataset_metadata_convert_to_float_internal(const char* str_value, float* result);

// Advanced type conversion with overflow protection and special value handling
bool llama_dataset_metadata_convert_to_int_safe_internal(const char* str_value, int64_t* result, bool* overflow_detected);
bool llama_dataset_metadata_convert_to_float_safe_internal(const char* str_value, float* result, bool* special_value_detected);

// Type checking utilities
bool llama_dataset_metadata_is_valid_int_string_internal(const char* str_value);
bool llama_dataset_metadata_is_valid_float_string_internal(const char* str_value);

// Default value handling for missing keys
int64_t llama_dataset_metadata_get_int_with_validation_internal(const struct llama_dataset* dataset, const char* key, int64_t default_value, bool* found);
float llama_dataset_metadata_get_float_with_validation_internal(const struct llama_dataset* dataset, const char* key, float default_value, bool* found);

// Format-specific key mapping and translation
enum dataset_format_type {
    DATASET_FORMAT_GGUF = 0,
    DATASET_FORMAT_TEXT = 1,
    DATASET_FORMAT_PARQUET = 2
};

const char* llama_dataset_metadata_map_key_for_format_internal(const char* key, enum dataset_format_type format_type);
bool llama_dataset_metadata_is_standard_key_internal(const char* key);
const char* llama_dataset_metadata_translate_key_internal(const char* key, enum dataset_format_type from_format, enum dataset_format_type to_format);

// Key validation with detailed error reporting
typedef enum {
    METADATA_KEY_VALID = 0,
    METADATA_KEY_NULL,
    METADATA_KEY_EMPTY,
    METADATA_KEY_TOO_LONG,
    METADATA_KEY_INVALID_CHARS,
    METADATA_KEY_INVALID_FORMAT
} metadata_key_validation_result;

metadata_key_validation_result llama_dataset_metadata_validate_key_detailed_internal(const char* key);
const char* llama_dataset_metadata_key_validation_result_to_string_internal(metadata_key_validation_result result);

// Enhanced metadata validation functions
bool llama_dataset_metadata_validate_metadata_set_internal(const struct llama_dataset* dataset, const char** required_keys, size_t num_required_keys);
bool llama_dataset_metadata_validate_consistency_internal(const struct llama_dataset* dataset);
bool llama_dataset_metadata_normalize_for_format_internal(const char* key, const char* value, enum dataset_format_type target_format, char* normalized_key, size_t key_buffer_size, char* normalized_value, size_t value_buffer_size);

#ifdef __cplusplus
}
#endif