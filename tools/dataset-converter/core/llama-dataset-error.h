#pragma once

#include <stdbool.h>
#include "llama-dataset.h" // For enum dataset_error definition

#ifdef __cplusplus
extern "C" {
#endif

// Internal error management functions (not exposed in public API)
void llama_dataset_error_set_internal(const char* message);
void llama_dataset_error_set_with_code_internal(enum dataset_error code, const char* message);
bool llama_dataset_error_has_error_internal(void);
void llama_dataset_error_clear_internal(void);
const char* llama_dataset_error_get_message_internal(void);
enum dataset_error llama_dataset_error_get_code_internal(void);

// Error message formatting with module context
void llama_dataset_error_set_with_context_internal(const char* module, const char* operation, const char* message);
void llama_dataset_error_set_with_context_and_code_internal(const char* module, const char* operation, enum dataset_error code, const char* message);

// Error code to string conversion
const char* llama_dataset_error_code_to_string_internal(enum dataset_error code);

// Advanced error message formatting with printf-style formatting
void llama_dataset_error_set_formatted_internal(enum dataset_error code, const char* format, ...);
void llama_dataset_error_set_formatted_with_context_internal(const char* module, const char* operation, enum dataset_error code, const char* format, ...);

// Error message inspection and utility functions
bool llama_dataset_error_is_truncated_internal(void);
size_t llama_dataset_error_get_message_length_internal(void);
void llama_dataset_error_append_context_internal(const char* context_info);

#ifdef __cplusplus
}
#endif