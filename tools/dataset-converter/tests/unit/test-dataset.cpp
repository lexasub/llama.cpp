#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>

#include "llama-dataset.h"
#include "llama-dataset-text.h"

// Simple test for the dataset interface
int main() {
    printf("Testing dataset interface...\n");

    // Test error handling with null path
    struct llama_dataset * dataset = from_gguf(nullptr);
    assert(dataset == nullptr);
    const char * error = llama_dataset_get_error();
    assert(error != nullptr);
    assert(strstr(error, "null") != nullptr);
    printf("✓ Null path error handling works\n");

    // Test error handling with non-existent file
    dataset = from_gguf("non_existent_file.gguf");
    assert(dataset == nullptr);
    error = llama_dataset_get_error();
    assert(error != nullptr);
    assert(strstr(error, "not found") != nullptr);
    printf("✓ File not found error handling works\n");

    // Test unimplemented functions
    dataset = from_txt("test.txt", nullptr);
    assert(dataset == nullptr);
    error = llama_dataset_get_error();
    assert(error != nullptr);
    assert(strstr(error, "not") != nullptr);
    printf("✓ Text loader placeholder works\n");

    dataset = from_parquet("test.parquet");
    assert(dataset == nullptr);
    error = llama_dataset_get_error();
    assert(error != nullptr);
    assert(strstr(error, "not") != nullptr);
    printf("✓ Parquet loader placeholder works\n");

    // Test legacy compatibility functions
    printf("\nTesting legacy compatibility functions...\n");

    // Create a simple test file
    std::ofstream test_file("test_compat.txt");
    test_file << "Test content for compatibility functions\n";
    test_file.close();

    // Test legacy functions with null path
    dataset = llama_dataset_load_gguf(nullptr, false);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    printf("✓ Legacy null path error handling works\n");

    // Test legacy functions with non-existent file
    dataset = llama_dataset_load_gguf("non_existent_file.gguf", false);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    printf("✓ Legacy file not found error handling works\n");

    // Test legacy text loading
    dataset = llama_dataset_load_text_internal("test_compat.txt", nullptr, false);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    printf("✓ Legacy text loader placeholder works\n");

    // Clean up test file
    remove("test_compat.txt");

    printf("All tests passed!\n");
    return 0;
}
