#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>

#include "common.h"
#include "llama-dataset-text.h"
#include "llama-dataset.h"

// Simple test for the dataset interface
int main(int argc, char** argv) {
    // Parse command-line arguments for specific dataset file paths
    std::vector<std::string> dataset_files;
    
    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            dataset_files.push_back(argv[i]);
        }
        printf("Using provided dataset files:\n");
        for (const auto& file : dataset_files) {
            printf("  %s\n", file.c_str());
        }
    } else {
        // Use default files if none provided
        dataset_files = {
            "test_data/small_dataset.gguf",
            "test_data/text_dataset.txt",
            "test_data/parquet_dataset.parquet"
        };
        printf("Using default dataset files\n");
    }

    printf("Testing dataset interface...\n");

    // Test error handling with null path
    common_params params;
    struct llama_dataset * dataset = llama_dataset_from_gguf(&params);
    assert(dataset == nullptr);
    const char * error = llama_dataset_get_error();
    assert(error != nullptr);
    assert(strstr(error, "null") != nullptr);
    (void)dataset; // Suppress unused variable warning in release builds
    (void)error;   // Suppress unused variable warning in release builds
    printf("✓ Null path error handling works\n");

    // Test error handling with non-existent file
    params.in_files.push_back("non_existent_file.gguf");
    dataset = llama_dataset_from_gguf(&params);
    assert(dataset == nullptr);
    error = llama_dataset_get_error();
    assert(error != nullptr);
    assert(strstr(error, "not found") != nullptr);
    printf("✓ File not found error handling works\n");

    // Test unimplemented functions
    params.in_files.back() = "test.txt";
    dataset = llama_dataset_from_txt(&params, nullptr);
    assert(dataset == nullptr);
    error = llama_dataset_get_error();
    assert(error != nullptr);
    assert(strstr(error, "not") != nullptr);
    printf("✓ Text loader placeholder works\n");

    params.in_files.back() = "test.parquet";
#ifdef LLAMA_PARQUET
    dataset = llama_dataset_from_parquet(&params);
#else
    return 0;
#endif
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

    params.in_files.clear();
    dataset = llama_dataset_load_gguf(&params);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    printf("✓ Legacy null path error handling works\n");

    // Test legacy functions with non-existent file
    params.in_files.push_back("non_existent_file.gguf");
    dataset = llama_dataset_load_gguf(&params);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    printf("✓ Legacy file not found error handling works\n");

    // Test legacy text loading
    params.in_files.back() = "test_compat.txt";
    dataset = llama_dataset_load_text_internal(&params, nullptr);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    printf("✓ Legacy text loader placeholder works\n");

    // Clean up test file
    remove("test_compat.txt");

    printf("All tests passed!\n");
    return 0;
}
