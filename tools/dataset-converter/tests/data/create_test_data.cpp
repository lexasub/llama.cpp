#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "ggml/include/ggml.h"
#include "gguf.h"
#include "include/llama.h"
#include "llama-dataset.h"

/**
 * Utility to create test data files for integration testing.
 * This creates small, known datasets in different formats for testing consistency.
 */

// Create a small GGUF file with known sequences
bool create_test_gguf_file(const char* path);
bool create_test_gguf_file(const char* path) {
    std::cout << "Creating test GGUF file: " << path << std::endl;

    // Define test sequences with known content
    std::vector<std::vector<int32_t>> test_sequences = {
        {1, 2, 3, 4, 5},           // sequence 0: "hello world" tokens
        {10, 20, 30},              // sequence 1: "test data" tokens
        {100, 200, 300, 400},      // sequence 2: "sample text here" tokens
        {1000, 2000}               // sequence 3: "short seq" tokens
    };

    // Create GGUF context
    struct gguf_context* ctx = gguf_init_empty();
    if (!ctx) {
        std::cerr << "Failed to create GGUF context" << std::endl;
        return false;
    }

    // Add metadata
    gguf_set_val_u64(ctx, TRAINING_SEQUENCE_COUNT, test_sequences.size());
    gguf_set_val_u64(ctx, TRAINING_MAX_LENGTH, 5); // Max sequence length
    gguf_set_val_str(ctx, TRAINING_FORMAT_SOURCE, "test");
    gguf_set_val_str(ctx, TRAINING_DATASET_DESCRIPTION, "Test dataset for integration tests");

    // Create ggml context for tensors
    struct ggml_init_params params = {
        1024 * 1024, // 1MB
        nullptr,
        false
    };
    struct ggml_context* ggml_ctx = ggml_init(params);
    if (!ggml_ctx) {
        std::cerr << "Failed to create ggml context" << std::endl;
        gguf_free(ctx);
        return false;
    }

    // Add each sequence as a tensor
    for (size_t i = 0; i < test_sequences.size(); i++) {
        const auto& seq = test_sequences[i];

        // Create tensor name
        char tensor_name[64];
        snprintf(tensor_name, sizeof(tensor_name), "sequence_%zu", i);

        // Create tensor
        struct ggml_tensor* tensor = ggml_new_tensor_1d(ggml_ctx, GGML_TYPE_I32, seq.size());
        if (!tensor) {
            std::cerr << "Failed to create tensor " << i << std::endl;
            ggml_free(ggml_ctx);
            gguf_free(ctx);
            return false;
        }

        // Set tensor name
        ggml_set_name(tensor, tensor_name);

        // Copy data to tensor
        memcpy(tensor->data, seq.data(), seq.size() * sizeof(int32_t));

        // Add tensor to GGUF
        gguf_add_tensor(ctx, tensor);
    }

    // Write GGUF file
    bool success = gguf_write_to_file(ctx, path, false);

    // Clean up
    ggml_free(ggml_ctx);
    gguf_free(ctx);

    if (success) {
        std::cout << "  Created GGUF file with " << test_sequences.size() << " sequences" << std::endl;
    } else {
        std::cerr << "  Failed to write GGUF file" << std::endl;
    }

    return success;
}

// Create equivalent text file with same sequences
bool create_test_text_file(const char* path);
bool create_test_text_file(const char* path) {
    std::cout << "Creating test text file: " << path << std::endl;

    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to create text file" << std::endl;
        return false;
    }

    // Write sequences as space-separated tokens (one sequence per line)
    file << "1 2 3 4 5\n";        // sequence 0
    file << "10 20 30\n";          // sequence 1
    file << "100 200 300 400\n";   // sequence 2
    file << "1000 2000\n";         // sequence 3

    file.close();

    std::cout << "  Created text file with 4 sequences" << std::endl;
    return true;
}

// Create large test files for performance testing
bool create_large_test_files();
bool create_large_test_files() {
    std::cout << "Creating large test files..." << std::endl;

    // Create large text file
    const char* large_text_path = "test_data/large_text_dataset.txt";
    std::ofstream large_text(large_text_path);
    if (!large_text.is_open()) {
        std::cerr << "Failed to create large text file" << std::endl;
        return false;
    }

    // Generate 1000 sequences with varying lengths
    for (int i = 0; i < 1000; i++) {
        int seq_len = 10 + (i % 20); // Length between 10-29
        for (int j = 0; j < seq_len; j++) {
            if (j > 0) large_text << " ";
            large_text << (i * 100 + j); // Generate unique tokens
        }
        large_text << "\n";
    }

    large_text.close();
    std::cout << "  Created large text file with 1000 sequences" << std::endl;
    return true;
}

// Create corrupted test files for error testing
bool create_corrupted_test_files();
bool create_corrupted_test_files() {
    std::cout << "Creating corrupted test files..." << std::endl;

    // Create corrupted GGUF file
    const char* corrupted_gguf_path = "test_data/corrupted_dataset.gguf";
    std::ofstream corrupted_gguf(corrupted_gguf_path, std::ios::binary);
    if (!corrupted_gguf.is_open()) {
        std::cerr << "Failed to create corrupted GGUF file" << std::endl;
        return false;
    }

    // Write invalid GGUF magic number
    const char bad_magic[] = "BADF";
    corrupted_gguf.write(bad_magic, 4);

    // Write some random data
    for (int i = 0; i < 100; i++) {
        uint8_t byte = i % 256;
        corrupted_gguf.write(reinterpret_cast<const char*>(&byte), 1);
    }

    corrupted_gguf.close();
    std::cout << "  Created corrupted GGUF file" << std::endl;

    // Create corrupted Parquet file
    const char* corrupted_parquet_path = "test_data/corrupted_dataset.parquet";
    std::ofstream corrupted_parquet(corrupted_parquet_path, std::ios::binary);
    if (!corrupted_parquet.is_open()) {
        std::cerr << "Failed to create corrupted Parquet file" << std::endl;
        return false;
    }

    // Write invalid Parquet magic number
    const char bad_parquet_magic[] = "BADF";
    corrupted_parquet.write(bad_parquet_magic, 4);

    // Write some random data
    for (int i = 0; i < 100; i++) {
        uint8_t byte = (i * 7) % 256;
        corrupted_parquet.write(reinterpret_cast<const char*>(&byte), 1);
    }

    corrupted_parquet.close();
    std::cout << "  Created corrupted Parquet file" << std::endl;

    return true;
}

int main() {
    std::cout << "=== Creating test data files for integration tests ===\n" << std::endl;

    // Create test_data directory if it doesn't exist
    system("mkdir -p test_data");
    system("mkdir -p tools/dataset-converter/tests/test_data");

    bool success = true;

    // Create small test files
    success &= create_test_gguf_file("test_data/small_dataset.gguf");
    success &= create_test_text_file("test_data/text_dataset.txt");

    // Copy to tests directory as well
    success &= create_test_gguf_file("tools/dataset-converter/tests/test_data/small_dataset.gguf");
    success &= create_test_text_file("tools/dataset-converter/tests/test_data/text_dataset.txt");

    // Create large test files
    success &= create_large_test_files();

    // Create corrupted test files
    success &= create_corrupted_test_files();

    if (success) {
        std::cout << "\n=== All test data files created successfully! ===" << std::endl;

        // List created files
        std::cout << "\nCreated files:" << std::endl;
        system("ls -la test_data/");
        std::cout << std::endl;
        system("ls -la tools/dataset-converter/tests/test_data/");

        return 0;
    } else {
        std::cerr << "\n=== Some test data files failed to create ===" << std::endl;
        return 1;
    }
}
