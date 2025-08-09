#include "llama-dataset-metadata.h"
#include "llama-dataset-error.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <cmath>

// Test utilities
void test_assert(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAILED: " << test_name << std::endl;
        exit(1);
    }
    std::cout << "PASSED: " << test_name << std::endl;
}

void clear_error_state() {
    llama_dataset_error_clear_internal();
}

// Test key validation functionality
void test_key_validation() {
    std::cout << "\n=== Testing Key Validation ===" << std::endl;
    
    // Test valid keys
    test_assert(llama_dataset_metadata_validate_key_internal("training.dataset.name"), 
                "Valid key with namespace");
    test_assert(llama_dataset_metadata_validate_key_internal("training.format.version"), 
                "Valid standard key");
    
    // Test invalid keys
    test_assert(!llama_dataset_metadata_validate_key_internal(nullptr), 
                "Null key should be invalid");
    test_assert(!llama_dataset_metadata_validate_key_internal(""), 
                "Empty key should be invalid");
    test_assert(!llama_dataset_metadata_validate_key_internal("toolong" + std::string(250, 'x')), 
                "Too long key should be invalid");
    test_assert(!llama_dataset_metadata_validate_key_internal("no_namespace"), 
                "Key without namespace should be invalid");
    test_assert(!llama_dataset_metadata_validate_key_internal("invalid\x01char"), 
                "Key with control character should be invalid");
}

void test_detailed_key_validation() {
    std::cout << "\n=== Testing Detailed Key Validation ===" << std::endl;
    
    // Test detailed validation results
    test_assert(llama_dataset_metadata_validate_key_detailed_internal("training.dataset.name") == METADATA_KEY_VALID,
                "Valid key returns METADATA_KEY_VALID");
    test_assert(llama_dataset_metadata_validate_key_detailed_internal(nullptr) == METADATA_KEY_NULL,
                "Null key returns METADATA_KEY_NULL");
    test_assert(llama_dataset_metadata_validate_key_detailed_internal("") == METADATA_KEY_EMPTY,
                "Empty key returns METADATA_KEY_EMPTY");
    test_assert(llama_dataset_metadata_validate_key_detailed_internal(std::string(300, 'x').c_str()) == METADATA_KEY_TOO_LONG,
                "Long key returns METADATA_KEY_TOO_LONG");
    test_assert(llama_dataset_metadata_validate_key_detailed_internal("invalid\x01char") == METADATA_KEY_INVALID_CHARS,
                "Key with control chars returns METADATA_KEY_INVALID_CHARS");
    test_assert(llama_dataset_metadata_validate_key_detailed_internal("no_namespace") == METADATA_KEY_INVALID_FORMAT,
                "Key without namespace returns METADATA_KEY_INVALID_FORMAT");
}

void test_key_validation_with_error() {
    std::cout << "\n=== Testing Key Validation with Error Integration ===" << std::endl;
    
    clear_error_state();
    
    // Test valid key doesn't set error
    test_assert(llama_dataset_metadata_validate_key_with_error_internal("training.dataset.name"),
                "Valid key validation succeeds");
    test_assert(!llama_dataset_error_has_error_internal(),
                "Valid key validation doesn't set error");
    
    // Test invalid key sets error
    test_assert(!llama_dataset_metadata_validate_key_with_error_internal(nullptr),
                "Null key validation fails");
    test_assert(llama_dataset_error_has_error_internal(),
                "Null key validation sets error");
    test_assert(strstr(llama_dataset_error_get_message_internal(), "Key is null") != nullptr,
                "Error message contains expected text");
    
    clear_error_state();
    
    // Test empty key
    test_assert(!llama_dataset_metadata_validate_key_with_error_internal(""),
                "Empty key validation fails");
    test_assert(llama_dataset_error_has_error_internal(),
                "Empty key validation sets error");
    test_assert(strstr(llama_dataset_error_get_message_internal(), "Key is empty") != nullptr,
                "Error message for empty key is correct");
    
    clear_error_state();
    
    // Test key without namespace
    test_assert(!llama_dataset_metadata_validate_key_with_error_internal("no_namespace"),
                "Key without namespace validation fails");
    test_assert(llama_dataset_error_has_error_internal(),
                "Key without namespace validation sets error");
    test_assert(strstr(llama_dataset_error_get_message_internal(), "namespace separator") != nullptr,
                "Error message mentions namespace separator");
}

void test_type_conversion_safe() {
    std::cout << "\n=== Testing Safe Type Conversion ===" << std::endl;
    
    // Test integer conversion
    int64_t int_result;
    bool overflow_detected;
    
    test_assert(llama_dataset_metadata_convert_to_int_safe_internal("123", &int_result, &overflow_detected),
                "Valid integer string converts successfully");
    test_assert(int_result == 123 && !overflow_detected,
                "Integer conversion result is correct");
    
    test_assert(llama_dataset_metadata_convert_to_int_safe_internal("  456  ", &int_result, &overflow_detected),
                "Integer with whitespace converts successfully");
    test_assert(int_result == 456 && !overflow_detected,
                "Integer with whitespace result is correct");
    
    test_assert(!llama_dataset_metadata_convert_to_int_safe_internal("not_a_number", &int_result, &overflow_detected),
                "Invalid integer string fails conversion");
    
    test_assert(!llama_dataset_metadata_convert_to_int_safe_internal("123abc", &int_result, &overflow_detected),
                "Integer with trailing chars fails conversion");
    
    // Test float conversion
    float float_result;
    bool special_value_detected;
    
    test_assert(llama_dataset_metadata_convert_to_float_safe_internal("3.14", &float_result, &special_value_detected),
                "Valid float string converts successfully");
    test_assert(std::abs(float_result - 3.14f) < 0.001f && !special_value_detected,
                "Float conversion result is correct");
    
    test_assert(llama_dataset_metadata_convert_to_float_safe_internal("inf", &float_result, &special_value_detected),
                "Infinity string converts successfully");
    test_assert(std::isinf(float_result) && special_value_detected,
                "Infinity conversion result is correct");
    
    test_assert(llama_dataset_metadata_convert_to_float_safe_internal("nan", &float_result, &special_value_detected),
                "NaN string converts successfully");
    test_assert(std::isnan(float_result) && special_value_detected,
                "NaN conversion result is correct");
    
    test_assert(!llama_dataset_metadata_convert_to_float_safe_internal("not_a_float", &float_result, &special_value_detected),
                "Invalid float string fails conversion");
}

void test_format_key_mapping() {
    std::cout << "\n=== Testing Format-Specific Key Mapping ===" << std::endl;
    
    clear_error_state();
    
    // Test standard key mapping (should return as-is)
    const char* mapped_key = llama_dataset_metadata_map_key_for_format_internal(
        TRAINING_DATASET_NAME, DATASET_FORMAT_GGUF);
    test_assert(mapped_key && strcmp(mapped_key, TRAINING_DATASET_NAME) == 0,
                "Standard key maps to itself for GGUF");
    test_assert(!llama_dataset_error_has_error_internal(),
                "Standard key mapping doesn't set error");
    
    // Test Parquet format mapping
    mapped_key = llama_dataset_metadata_map_key_for_format_internal("name", DATASET_FORMAT_PARQUET);
    test_assert(mapped_key && strcmp(mapped_key, TRAINING_DATASET_NAME) == 0,
                "Parquet 'name' maps to standard dataset name");
    
    mapped_key = llama_dataset_metadata_map_key_for_format_internal("description", DATASET_FORMAT_PARQUET);
    test_assert(mapped_key && strcmp(mapped_key, TRAINING_DATASET_DESCRIPTION) == 0,
                "Parquet 'description' maps to standard description");
    
    // Test invalid parameters
    mapped_key = llama_dataset_metadata_map_key_for_format_internal(nullptr, DATASET_FORMAT_GGUF);
    test_assert(mapped_key == nullptr,
                "Null key returns null");
    test_assert(llama_dataset_error_has_error_internal(),
                "Null key sets error");
    
    clear_error_state();
    
    // Test invalid key
    mapped_key = llama_dataset_metadata_map_key_for_format_internal("invalid_key", DATASET_FORMAT_GGUF);
    test_assert(mapped_key == nullptr,
                "Invalid key returns null");
    test_assert(llama_dataset_error_has_error_internal(),
                "Invalid key sets error");
}

void test_key_translation() {
    std::cout << "\n=== Testing Key Translation Between Formats ===" << std::endl;
    
    clear_error_state();
    
    // Test translation between same formats (should return as-is)
    const char* translated_key = llama_dataset_metadata_translate_key_internal(
        "training.dataset.name", DATASET_FORMAT_GGUF, DATASET_FORMAT_GGUF);
    test_assert(translated_key && strcmp(translated_key, "training.dataset.name") == 0,
                "Same format translation returns original key");
    
    // Test standard key translation (should return as-is)
    translated_key = llama_dataset_metadata_translate_key_internal(
        TRAINING_DATASET_NAME, DATASET_FORMAT_GGUF, DATASET_FORMAT_PARQUET);
    test_assert(translated_key && strcmp(translated_key, TRAINING_DATASET_NAME) == 0,
                "Standard key translation returns original key");
    
    // Test null key
    translated_key = llama_dataset_metadata_translate_key_internal(
        nullptr, DATASET_FORMAT_GGUF, DATASET_FORMAT_PARQUET);
    test_assert(translated_key == nullptr,
                "Null key translation returns null");
    test_assert(llama_dataset_error_has_error_internal(),
                "Null key translation sets error");
}

void test_standard_key_recognition() {
    std::cout << "\n=== Testing Standard Key Recognition ===" << std::endl;
    
    // Test standard keys
    test_assert(llama_dataset_metadata_is_standard_key_internal(TRAINING_FORMAT_VERSION),
                "TRAINING_FORMAT_VERSION is recognized as standard");
    test_assert(llama_dataset_metadata_is_standard_key_internal(TRAINING_DATASET_NAME),
                "TRAINING_DATASET_NAME is recognized as standard");
    test_assert(llama_dataset_metadata_is_standard_key_internal(TRAINING_SEQUENCE_COUNT),
                "TRAINING_SEQUENCE_COUNT is recognized as standard");
    
    // Test non-standard keys
    test_assert(!llama_dataset_metadata_is_standard_key_internal("custom.key.name"),
                "Custom key is not recognized as standard");
    test_assert(!llama_dataset_metadata_is_standard_key_internal(nullptr),
                "Null key is not recognized as standard");
    test_assert(!llama_dataset_metadata_is_standard_key_internal(""),
                "Empty key is not recognized as standard");
}

void test_validation_result_strings() {
    std::cout << "\n=== Testing Validation Result String Conversion ===" << std::endl;
    
    // Test all validation result strings
    test_assert(strcmp(llama_dataset_metadata_key_validation_result_to_string_internal(METADATA_KEY_VALID), 
                      "Key is valid") == 0,
                "METADATA_KEY_VALID string is correct");
    test_assert(strcmp(llama_dataset_metadata_key_validation_result_to_string_internal(METADATA_KEY_NULL), 
                      "Key is null") == 0,
                "METADATA_KEY_NULL string is correct");
    test_assert(strcmp(llama_dataset_metadata_key_validation_result_to_string_internal(METADATA_KEY_EMPTY), 
                      "Key is empty") == 0,
                "METADATA_KEY_EMPTY string is correct");
    test_assert(strcmp(llama_dataset_metadata_key_validation_result_to_string_internal(METADATA_KEY_TOO_LONG), 
                      "Key is too long (max 255 characters)") == 0,
                "METADATA_KEY_TOO_LONG string is correct");
    test_assert(strcmp(llama_dataset_metadata_key_validation_result_to_string_internal(METADATA_KEY_INVALID_CHARS), 
                      "Key contains invalid characters") == 0,
                "METADATA_KEY_INVALID_CHARS string is correct");
    test_assert(strcmp(llama_dataset_metadata_key_validation_result_to_string_internal(METADATA_KEY_INVALID_FORMAT), 
                      "Key format is invalid (should contain namespace separator '.')") == 0,
                "METADATA_KEY_INVALID_FORMAT string is correct");
}

void test_type_checking_utilities() {
    std::cout << "\n=== Testing Type Checking Utilities ===" << std::endl;
    
    // Test integer string validation
    test_assert(llama_dataset_metadata_is_valid_int_string_internal("123"),
                "Valid integer string is recognized");
    test_assert(llama_dataset_metadata_is_valid_int_string_internal("-456"),
                "Negative integer string is recognized");
    test_assert(llama_dataset_metadata_is_valid_int_string_internal("  789  "),
                "Integer string with whitespace is recognized");
    test_assert(!llama_dataset_metadata_is_valid_int_string_internal("123.45"),
                "Float string is not recognized as integer");
    test_assert(!llama_dataset_metadata_is_valid_int_string_internal("not_a_number"),
                "Non-numeric string is not recognized as integer");
    test_assert(!llama_dataset_metadata_is_valid_int_string_internal(nullptr),
                "Null string is not recognized as integer");
    
    // Test float string validation
    test_assert(llama_dataset_metadata_is_valid_float_string_internal("3.14"),
                "Valid float string is recognized");
    test_assert(llama_dataset_metadata_is_valid_float_string_internal("123"),
                "Integer string is recognized as valid float");
    test_assert(llama_dataset_metadata_is_valid_float_string_internal("inf"),
                "Infinity string is recognized as valid float");
    test_assert(llama_dataset_metadata_is_valid_float_string_internal("nan"),
                "NaN string is recognized as valid float");
    test_assert(!llama_dataset_metadata_is_valid_float_string_internal("not_a_number"),
                "Non-numeric string is not recognized as float");
    test_assert(!llama_dataset_metadata_is_valid_float_string_internal(nullptr),
                "Null string is not recognized as float");
}

int main() {
    std::cout << "=== Metadata Validation and Error Integration Tests ===" << std::endl;
    
    try {
        test_key_validation();
        test_detailed_key_validation();
        test_key_validation_with_error();
        test_type_conversion_safe();
        test_format_key_mapping();
        test_key_translation();
        test_standard_key_recognition();
        test_validation_result_strings();
        test_type_checking_utilities();
        
        std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Test failed with unknown exception" << std::endl;
        return 1;
    }
}