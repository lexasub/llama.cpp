#include <iostream>
#include <cassert>
#include <cstring>
#include <string>

// Include the error module
#include "llama-dataset.h"
#include "llama-dataset-error.h"

/**
 * @brief Comprehensive test for error message formatting functionality.
 * 
 * This test validates all aspects of error message formatting including:
 * - Error code to string conversion
 * - Printf-style formatted error messages
 * - Context-aware error formatting
 * - Message truncation and safety checks
 * - Error message inspection utilities
 */

// Test helper to check if a string contains a substring
bool contains(const char* haystack, const char* needle) {
    return strstr(haystack, needle) != nullptr;
}

void test_error_code_to_string() {
    std::cout << "Test 1: Error code to string conversion..." << std::endl;
    
    // Test all error codes
    assert(strcmp(llama_dataset_error_code_to_string_internal(DATASET_SUCCESS), "Success") == 0);
    assert(strcmp(llama_dataset_error_code_to_string_internal(DATASET_ERROR_FILE_NOT_FOUND), "File not found") == 0);
    assert(strcmp(llama_dataset_error_code_to_string_internal(DATASET_ERROR_INVALID_FORMAT), "Invalid format") == 0);
    assert(strcmp(llama_dataset_error_code_to_string_internal(DATASET_ERROR_MEMORY_ALLOCATION), "Memory allocation failed") == 0);
    assert(strcmp(llama_dataset_error_code_to_string_internal(DATASET_ERROR_TOKENIZATION_FAILED), "Tokenization failed") == 0);
    assert(strcmp(llama_dataset_error_code_to_string_internal(DATASET_ERROR_STREAMING_NOT_SUPPORTED), "Streaming not supported") == 0);
    assert(strcmp(llama_dataset_error_code_to_string_internal(DATASET_ERROR_INVALID_PARAMETER), "Invalid parameter") == 0);
    assert(strcmp(llama_dataset_error_code_to_string_internal(DATASET_ERROR_CONTEXT_CREATION_FAILED), "Context creation failed") == 0);
    assert(strcmp(llama_dataset_error_code_to_string_internal(DATASET_ERROR_IO_ERROR), "I/O error") == 0);
    assert(strcmp(llama_dataset_error_code_to_string_internal(DATASET_ERROR_UNKNOWN), "Unknown error") == 0);
    
    // Test invalid error code (should return "Unknown error")
    assert(strcmp(llama_dataset_error_code_to_string_internal(static_cast<enum dataset_error>(999)), "Unknown error") == 0);
    
    std::cout << "✅ Error code to string conversion works correctly" << std::endl;
}

void test_formatted_error_messages() {
    std::cout << "Test 2: Formatted error messages..." << std::endl;
    
    // Clear any existing error
    llama_dataset_error_clear_internal();
    
    // Test basic formatted message
    llama_dataset_error_set_formatted_internal(DATASET_ERROR_FILE_NOT_FOUND, "File '%s' not found at line %d", "test.txt", 42);
    assert(llama_dataset_error_has_error_internal());
    assert(llama_dataset_error_get_code_internal() == DATASET_ERROR_FILE_NOT_FOUND);
    
    const char* message = llama_dataset_error_get_message_internal();
    assert(contains(message, "test.txt"));
    assert(contains(message, "42"));
    assert(contains(message, "not found"));
    
    std::cout << "✅ Basic formatted messages work correctly" << std::endl;
}

void test_formatted_error_with_context() {
    std::cout << "Test 3: Formatted error messages with context..." << std::endl;
    
    // Clear any existing error
    llama_dataset_error_clear_internal();
    
    // Test formatted message with context
    llama_dataset_error_set_formatted_with_context_internal(
        "gguf_loader", "load_sequence", 
        DATASET_ERROR_INVALID_FORMAT, 
        "Invalid sequence at index %d, expected %d tokens but got %d", 
        123, 512, 256
    );
    
    assert(llama_dataset_error_has_error_internal());
    assert(llama_dataset_error_get_code_internal() == DATASET_ERROR_INVALID_FORMAT);
    
    const char* message = llama_dataset_error_get_message_internal();
    assert(contains(message, "[gguf_loader::load_sequence]"));
    assert(contains(message, "123"));
    assert(contains(message, "512"));
    assert(contains(message, "256"));
    assert(contains(message, "Invalid sequence"));
    
    std::cout << "✅ Formatted messages with context work correctly" << std::endl;
}

void test_message_truncation() {
    std::cout << "Test 4: Message truncation and safety..." << std::endl;
    
    // Clear any existing error
    llama_dataset_error_clear_internal();
    
    // Create a very long error message that should be truncated
    std::string long_message(2000, 'A'); // 2000 'A' characters
    llama_dataset_error_set_formatted_internal(DATASET_ERROR_UNKNOWN, "Long message: %s", long_message.c_str());
    
    const char* message = llama_dataset_error_get_message_internal();
    size_t message_len = llama_dataset_error_get_message_length_internal();
    
    // Message should be truncated to fit in buffer (1024 chars including null terminator)
    assert(message_len < 1024);
    assert(message_len > 0);
    
    // Should end with truncation indicator
    assert(llama_dataset_error_is_truncated_internal());
    assert(contains(message, "..."));
    
    std::cout << "✅ Message truncation works correctly" << std::endl;
}

void test_context_truncation() {
    std::cout << "Test 5: Context-aware truncation..." << std::endl;
    
    // Clear any existing error
    llama_dataset_error_clear_internal();
    
    // Create a long message with context that should be truncated
    std::string long_message(2000, 'B'); // 2000 'B' characters
    llama_dataset_error_set_formatted_with_context_internal(
        "very_long_module_name", "very_long_operation_name",
        DATASET_ERROR_IO_ERROR,
        "Very long error message: %s", long_message.c_str()
    );
    
    const char* message = llama_dataset_error_get_message_internal();
    
    // Should contain context prefix
    assert(contains(message, "[very_long_module_name::very_long_operation_name]"));
    
    // Should be truncated
    assert(llama_dataset_error_is_truncated_internal());
    
    // Should not exceed buffer size
    assert(llama_dataset_error_get_message_length_internal() < 1024);
    
    std::cout << "✅ Context-aware truncation works correctly" << std::endl;
}

void test_context_appending() {
    std::cout << "Test 6: Context appending..." << std::endl;
    
    // Clear any existing error
    llama_dataset_error_clear_internal();
    
    // Set initial error
    llama_dataset_error_set_internal("Initial error");
    
    // Append context information
    llama_dataset_error_append_context_internal("(file: test.gguf)");
    
    const char* message = llama_dataset_error_get_message_internal();
    assert(contains(message, "Initial error"));
    assert(contains(message, "(file: test.gguf)"));
    
    // Append more context
    llama_dataset_error_append_context_internal("(line: 42)");
    
    message = llama_dataset_error_get_message_internal();
    assert(contains(message, "Initial error"));
    assert(contains(message, "(file: test.gguf)"));
    assert(contains(message, "(line: 42)"));
    
    std::cout << "✅ Context appending works correctly" << std::endl;
}

void test_edge_cases() {
    std::cout << "Test 7: Edge cases and safety..." << std::endl;
    
    // Clear any existing error
    llama_dataset_error_clear_internal();
    
    // Test null parameters
    llama_dataset_error_set_formatted_internal(DATASET_ERROR_UNKNOWN, nullptr);
    assert(strcmp(llama_dataset_error_get_message_internal(), "") == 0);
    
    llama_dataset_error_set_formatted_with_context_internal(nullptr, nullptr, DATASET_ERROR_UNKNOWN, nullptr);
    assert(strcmp(llama_dataset_error_get_message_internal(), "") == 0);
    
    llama_dataset_error_set_formatted_with_context_internal("module", "op", DATASET_ERROR_UNKNOWN, nullptr);
    assert(strcmp(llama_dataset_error_get_message_internal(), "") == 0);
    
    // Test empty context appending
    llama_dataset_error_set_internal("Test message");
    llama_dataset_error_append_context_internal(nullptr);
    llama_dataset_error_append_context_internal("");
    assert(strcmp(llama_dataset_error_get_message_internal(), "Test message") == 0);
    
    // Test appending to empty message
    llama_dataset_error_clear_internal();
    llama_dataset_error_append_context_internal("Context only");
    assert(contains(llama_dataset_error_get_message_internal(), "Context only"));
    
    std::cout << "✅ Edge cases handled safely" << std::endl;
}

void test_internal_api_integration() {
    std::cout << "Test 8: Internal API integration..." << std::endl;
    
    // Test that the internal API function works
    const char* success_str = llama_dataset_error_code_to_string_internal(DATASET_SUCCESS);
    assert(strcmp(success_str, "Success") == 0);
    
    const char* file_not_found_str = llama_dataset_error_code_to_string_internal(DATASET_ERROR_FILE_NOT_FOUND);
    assert(strcmp(file_not_found_str, "File not found") == 0);
    
    std::cout << "✅ Internal API integration works correctly" << std::endl;
}

int main() {
    std::cout << "Testing comprehensive error message formatting..." << std::endl;
    std::cout << "=============================================" << std::endl;
    
    test_error_code_to_string();
    test_formatted_error_messages();
    test_formatted_error_with_context();
    test_message_truncation();
    test_context_truncation();
    test_context_appending();
    test_edge_cases();
    test_internal_api_integration();
    
    std::cout << std::endl << "🎉 All error formatting tests passed!" << std::endl;
    std::cout << "Comprehensive error message formatting has been successfully implemented." << std::endl;
    std::cout << "Features implemented:" << std::endl;
    std::cout << "  ✅ Error code to string conversion" << std::endl;
    std::cout << "  ✅ Printf-style formatted error messages" << std::endl;
    std::cout << "  ✅ Context-aware error formatting" << std::endl;
    std::cout << "  ✅ Message truncation and safety checks" << std::endl;
    std::cout << "  ✅ Error message inspection utilities" << std::endl;
    std::cout << "  ✅ Context appending functionality" << std::endl;
    std::cout << "  ✅ Internal API integration" << std::endl;
    
    return 0;
}