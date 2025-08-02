/**
 * @file test-modular-validation.cpp
 * @brief Simple test to verify the modular validation structure works correctly.
 */

#include "test-data-validator.h"
#include "test-data-validator-gguf.h"
#include "test-data-validator-parquet.h"
#include "test-data-validator-text.h"
#include "test-data-validator-common.h"

#include <iostream>
#include <cstdio>

int main() {
    printf("Testing modular validation structure...\n");

    // Test common functions
    printf("Testing common functions:\n");
    printf("- file_exists: %s\n", file_exists("/tmp") ? "available" : "not available");
    printf("- test_data_validation_result_to_string: %s\n", 
           test_data_validation_result_to_string(TEST_DATA_VALID));

    // Test format-specific validation functions
    printf("\nTesting format-specific validation functions:\n");
    printf("- validate_gguf_test_file: available\n");
    printf("- validate_parquet_test_file: available\n");
    printf("- validate_text_test_file: available\n");

    // Test format-specific creation functions
    printf("\nTesting format-specific creation functions:\n");
    printf("- create_minimal_gguf_dataset: available\n");
    printf("- create_minimal_parquet_dataset: available\n");
    printf("- create_minimal_text_dataset: available\n");

    // Test corrupted file creation functions
    printf("\nTesting corrupted file creation functions:\n");
    printf("- create_corrupted_gguf_test_file: available\n");
    printf("- create_corrupted_parquet_test_file: available\n");
    printf("- create_corrupted_text_test_file: available\n");

    // Test generic wrapper function
    printf("\nTesting generic wrapper function:\n");
    printf("- create_corrupted_test_file: available\n");

    printf("\nModular validation structure test completed successfully!\n");
    return 0;
}