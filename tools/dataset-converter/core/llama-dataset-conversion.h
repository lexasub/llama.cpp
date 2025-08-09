#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct llama_dataset;

// Dataset conversion interface
bool llama_dataset_conversion_to_gguf(struct llama_dataset* dataset, const char* output_filename);

// Format transformation utilities
bool llama_dataset_conversion_transform_format(struct llama_dataset* dataset, int target_format);
bool llama_dataset_conversion_preserve_metadata(struct llama_dataset* source, struct llama_dataset* target);

// Conversion validation and integrity checking
bool llama_dataset_conversion_validate_conversion(const struct llama_dataset* source, const struct llama_dataset* target);
bool llama_dataset_conversion_check_integrity(const struct llama_dataset* dataset, const char* filename);

// Conversion progress tracking and error reporting
typedef struct {
    size_t total_sequences;
    size_t processed_sequences;
    double progress_percentage;
    bool has_error;
    char error_message[256];
} llama_dataset_conversion_progress;

bool llama_dataset_conversion_get_progress(const struct llama_dataset* dataset, llama_dataset_conversion_progress* progress);
void llama_dataset_conversion_reset_progress(struct llama_dataset* dataset);

// Format-specific optimization
bool llama_dataset_conversion_optimize_for_format(struct llama_dataset* dataset, int target_format);
bool llama_dataset_conversion_apply_format_specific_settings(struct llama_dataset* dataset, int target_format);

#ifdef __cplusplus
}
#endif