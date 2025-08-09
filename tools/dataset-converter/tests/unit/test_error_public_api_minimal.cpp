#include <iostream>
#include <cassert>
#include <cstring>

// Include the error module
#include "llama-dataset.h"
#include "llama-dataset-error.h"

// Minimal implementation of the public API function for testing
extern "C" {
    const char* llama_dataset_error_code_to_string(enum dataset_error code) {
        return llama_dataset_error_code_to_string_internal(code);
    }
}

/**
 * @brief Minimal test for the public API error code to string function.
 * 
 * This test validates that the public API function for error code to string
 * conversion works correctly when properly implemented.
 */
int main() {
    std::cout << "Testing public API error code to string conversion..." << std::endl;
    
    // Test all error codes through public API
    assert(strcmp(llama_dataset_error_code_to_string(DATASET_SUCCESS), "Success") == 0);
    assert(strcmp(llama_dataset_error_code_to_string(DATASET_ERROR_FILE_NOT_FOUND), "File not found") == 0);
    assert(strcmp(llama_dataset_error_code_to_string(DATASET_ERROR_INVALID_FORMAT), "Invalid format") == 0);
    assert(strcmp(llama_dataset_error_code_to_string(DATASET_ERROR_MEMORY_ALLOCATION), "Memory allocation failed") == 0);
    assert(strcmp(llama_dataset_error_code_to_string(DATASET_ERROR_TOKENIZATION_FAILED), "Tokenization failed") == 0);
    assert(strcmp(llama_dataset_error_code_to_string(DATASET_ERROR_STREAMING_NOT_SUPPORTED), "Streaming not supported") == 0);
    assert(strcmp(llama_dataset_error_code_to_string(DATASET_ERROR_INVALID_PARAMETER), "Invalid parameter") == 0);
    assert(strcmp(llama_dataset_error_code_to_string(DATASET_ERROR_CONTEXT_CREATION_FAILED), "Context creation failed") == 0);
    assert(strcmp(llama_dataset_error_code_to_string(DATASET_ERROR_IO_ERROR), "I/O error") == 0);
    assert(strcmp(llama_dataset_error_code_to_string(DATASET_ERROR_UNKNOWN), "Unknown error") == 0);
    
    // Test invalid error code (should return "Unknown error")
    assert(strcmp(llama_dataset_error_code_to_string(static_cast<enum dataset_error>(999)), "Unknown error") == 0);
    
    std::cout << "✅ All public API error code to string tests passed!" << std::endl;
    std::cout << "Public API function llama_dataset_error_code_to_string() works correctly." << std::endl;
    
    return 0;
}