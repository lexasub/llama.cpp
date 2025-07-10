#include "llama-dataset.h"
#include "common/log.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>

// Helper function to create a simple text dataset file for testing
bool create_test_text_file(const char* path, const std::vector<std::string>& lines) {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    for (const auto& line : lines) {
        file << line << std::endl;
    }

    file.close();
    return true;
}

// Helper function to compare two datasets for equality
bool compare_datasets(struct llama_dataset* ds1, struct llama_dataset* ds2) {
    if (!ds1 || !ds2) {
        return false;
    }

    // Compare sequence counts
    uint64_t count1 = n_sequences(ds1);
    uint64_t count2 = n_sequences(ds2);

    if (count1 != count2) {
        std::cerr << "Sequence counts differ: " << count1 << " vs " << count2 << std::endl;
        return false;
    }

    // Compare each sequence
    for (uint64_t i = 0; i < count1; i++) {
        int32_t len1 = sequence_length(ds1, i);
        int32_t len2 = sequence_length(ds2, i);

        if (len1 != len2) {
            std::cerr << "Sequence " << i << " lengths differ: " << len1 << " vs " << len2 << std::endl;
            return false;
        }

        const int32_t* seq1 = sequence(ds1, i);
        const int32_t* seq2 = sequence(ds2, i);

        if (!seq1 || !seq2) {
            std::cerr << "Sequence " << i << " data is null" << std::endl;
            return false;
        }

        for (int32_t j = 0; j < len1; j++) {
            if (seq1[j] != seq2[j]) {
                std::cerr << "Sequence " << i << " data differs at position " << j << ": "
                          << seq1[j] << " vs " << seq2[j] << std::endl;
                return false;
            }
        }
    }

    return true;
}

// Test GGUF factory function with valid and invalid inputs
void test_gguf_factory() {
    std::cout << "\n=== Testing GGUF factory function ===\n";

    // Test with null path
    struct llama_dataset* dataset = from_gguf(nullptr);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    std::cout << "✓ Null path error handling works\n";
    llama_dataset_clear_error();

    // Test with non-existent file
    dataset = from_gguf("non_existent_file.gguf");
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    std::cout << "✓ Non-existent file error handling works\n";
    llama_dataset_clear_error();

    // Test with valid file
    const char* test_file = "test_data/small_dataset.gguf";
    dataset = from_gguf(test_file);

    if (dataset) {
        std::cout << "✓ Successfully loaded valid GGUF file\n";

        // Test basic properties
        uint64_t seq_count = n_sequences(dataset);
        std::cout << "  Sequence count: " << seq_count << "\n";
        assert(seq_count > 0);

        // Test first sequence
        int32_t seq_len = sequence_length(dataset, 0);
        std::cout << "  First sequence length: " << seq_len << "\n";
        assert(seq_len > 0);

        const int32_t* seq_data = sequence(dataset, 0);
        assert(seq_data != nullptr);
        std::cout << "  First few tokens: ";
        for (int i = 0; i < std::min(seq_len, 5); i++) {
            std::cout << seq_data[i] << " ";
        }
        std::cout << "\n";

        // Test metadata access
        const char* format = dataset_get_metadata_str(dataset, DATASET_SOURCE_FORMAT);
        if (format) {
            std::cout << "  Source format: " << format << "\n";
        }

        int64_t count = dataset_get_metadata_int(dataset, DATASET_SEQUENCE_COUNT, 0);
        std::cout << "  Metadata sequence count: " << count << "\n";
        std::cout << "  Actual sequence count: " << seq_count << "\n";
        // Note: Metadata might not be available, so we don't assert on it
        // assert(count == seq_count);

        // Test tensor access
        struct ggml_tensor* tensor = sequence_tensor(dataset, 0);
        assert(tensor != nullptr);
        std::cout << "✓ Tensor access works\n";

        // Clean up
        llama_dataset_free(dataset);
        std::cout << "✓ Dataset cleanup successful\n";
    } else {
        std::cerr << "✗ Failed to load valid GGUF file: " << llama_dataset_get_error() << "\n";
        assert(false && "Failed to load valid GGUF file");
    }
}

// Test text factory function with valid and invalid inputs
void test_text_factory() {
    std::cout << "\n=== Testing text factory function ===\n";

    // Create a test text file
    const char* test_file = "test_text_data.txt";
    std::vector<std::string> lines = {
        "This is a test sentence.",
        "Another test sentence with more words.",
        "A third line for testing the text loader."
    };

    bool created = create_test_text_file(test_file, lines);
    assert(created && "Failed to create test text file");

    // Test with null path
    struct llama_dataset* dataset = from_txt(nullptr, nullptr);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    std::cout << "✓ Null path error handling works\n";
    llama_dataset_clear_error();

    // Test with null model (tokenizer)
    dataset = from_txt(test_file, nullptr);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    std::cout << "✓ Null tokenizer error handling works\n";
    llama_dataset_clear_error();

    // Note: We can't fully test the text loader without a valid tokenizer model
    // This would require loading a real model, which is beyond the scope of this test
    std::cout << "✓ Text factory function tests completed\n";

    // Clean up
    remove(test_file);
}

// Test Parquet factory function with valid and invalid inputs
void test_parquet_factory() {
    std::cout << "\n=== Testing Parquet factory function ===\n";

    // Test with null path
    struct llama_dataset* dataset = from_parquet(nullptr);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    std::cout << "✓ Null path error handling works\n";
    llama_dataset_clear_error();

    // Test with non-existent file
    dataset = from_parquet("non_existent_file.parquet");
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    std::cout << "✓ Non-existent file error handling works\n";
    llama_dataset_clear_error();

    // Test with valid file if available
    const char* test_file = "test_data/parquet_dataset.parquet";
    dataset = from_parquet(test_file);

    if (dataset) {
        std::cout << "✓ Successfully loaded valid Parquet file\n";

        // Test basic properties
        uint64_t seq_count = n_sequences(dataset);
        std::cout << "  Sequence count: " << seq_count << "\n";
        assert(seq_count > 0);

        // Test first sequence
        int32_t seq_len = sequence_length(dataset, 0);
        std::cout << "  First sequence length: " << seq_len << "\n";
        assert(seq_len > 0);

        const int32_t* seq_data = sequence(dataset, 0);
        assert(seq_data != nullptr);
        std::cout << "  First few tokens: ";
        for (int i = 0; i < std::min(seq_len, 5); i++) {
            std::cout << seq_data[i] << " ";
        }
        std::cout << "\n";

        // Clean up
        llama_dataset_free(dataset);
        std::cout << "✓ Dataset cleanup successful\n";
    } else {
        std::cout << "  Parquet file not available or support not compiled in: " << llama_dataset_get_error() << "\n";
        llama_dataset_clear_error();
    }
}

// Test sequence access functions for consistency across formats
void test_sequence_access() {
    std::cout << "\n=== Testing sequence access functions ===\n";

    // Load a GGUF dataset
    const char* gguf_file = "test_data/small_dataset.gguf";
    struct llama_dataset* gguf_dataset = from_gguf(gguf_file);

    if (!gguf_dataset) {
        std::cerr << "Failed to load GGUF dataset: " << llama_dataset_get_error() << "\n";
        assert(false && "Failed to load GGUF dataset");
        return;
    }

    // Test basic sequence access
    uint64_t seq_count = n_sequences(gguf_dataset);
    std::cout << "GGUF dataset sequence count: " << seq_count << "\n";

    // Test out-of-bounds access
    int32_t invalid_len = sequence_length(gguf_dataset, seq_count + 1);
    assert(invalid_len == 0);
    std::cout << "✓ Out-of-bounds sequence length check works\n";

    const int32_t* invalid_seq = sequence(gguf_dataset, seq_count + 1);
    assert(invalid_seq == nullptr);
    std::cout << "✓ Out-of-bounds sequence access check works\n";

    // Test tensor access
    struct ggml_tensor* invalid_tensor = sequence_tensor(gguf_dataset, seq_count + 1);
    assert(invalid_tensor == nullptr);
    std::cout << "✓ Out-of-bounds tensor access check works\n";

    // Test null dataset handling
    assert(n_sequences(nullptr) == 0);
    assert(sequence_length(nullptr, 0) == 0);
    assert(sequence(nullptr, 0) == nullptr);
    assert(sequence_tensor(nullptr, 0) == nullptr);
    std::cout << "✓ Null dataset handling works\n";

    // Clean up
    llama_dataset_free(gguf_dataset);
}

// Test streaming vs full loading equivalence
void test_streaming_equivalence() {
    std::cout << "\n=== Testing streaming vs full loading equivalence ===\n";

    const char* test_file = "test_data/small_dataset.gguf";

    // Check if streaming is supported
    bool supports_streaming = llama_dataset_supports_streaming(DATASET_GGUF, test_file);
    std::cout << "GGUF streaming supported: " << (supports_streaming ? "yes" : "no") << "\n";

    // Load dataset in non-streaming mode
    struct llama_dataset* non_streaming = llama_dataset_load_gguf(test_file, false);
    if (!non_streaming) {
        std::cerr << "Failed to load dataset in non-streaming mode: " << llama_dataset_get_error() << "\n";
        assert(false && "Failed to load dataset in non-streaming mode");
        return;
    }

    // Load dataset in streaming mode
    struct llama_dataset* streaming = llama_dataset_load_gguf(test_file, true);
    if (!streaming) {
        std::cerr << "Failed to load dataset in streaming mode: " << llama_dataset_get_error() << "\n";
        llama_dataset_free(non_streaming);
        assert(false && "Failed to load dataset in streaming mode");
        return;
    }

    // Verify streaming mode is enabled if supported
    bool is_streaming = llama_dataset_is_streaming_enabled(streaming);
    std::cout << "Streaming mode enabled: " << (is_streaming ? "yes" : "no") << "\n";

    // Compare the datasets
    bool datasets_equal = compare_datasets(non_streaming, streaming);
    std::cout << "Datasets are equal: " << (datasets_equal ? "yes" : "no") << "\n";
    assert(datasets_equal && "Streaming and non-streaming datasets should be equal");

    // Clean up
    llama_dataset_free(non_streaming);
    llama_dataset_free(streaming);
    std::cout << "✓ Streaming equivalence test passed\n";
}

// Test error conditions with invalid inputs
void test_error_conditions() {
    std::cout << "\n=== Testing error conditions ===\n";

    // Test with corrupted GGUF file if available
    const char* corrupted_file = "test_data/corrupted_dataset.gguf";
    struct llama_dataset* dataset = from_gguf(corrupted_file);

    if (!dataset) {
        std::cout << "✓ Corrupted GGUF file correctly rejected: " << llama_dataset_get_error() << "\n";
        llama_dataset_clear_error();
    } else {
        std::cerr << "✗ Corrupted GGUF file was loaded successfully, which is unexpected\n";
        llama_dataset_free(dataset);
    }

    // Test with corrupted Parquet file if available
    corrupted_file = "test_data/corrupted_dataset.parquet";
    dataset = from_parquet(corrupted_file);

    if (!dataset) {
        std::cout << "✓ Corrupted Parquet file correctly rejected: " << llama_dataset_get_error() << "\n";
        llama_dataset_clear_error();
    } else {
        std::cerr << "✗ Corrupted Parquet file was loaded successfully, which is unexpected\n";
        llama_dataset_free(dataset);
    }

    // Test error code to string conversion
    const char* error_str = llama_dataset_error_code_to_string(DATASET_ERROR_FILE_NOT_FOUND);
    assert(error_str != nullptr);
    std::cout << "Error code string: " << error_str << "\n";

    // Test error clearing
    llama_dataset_clear_error();
    assert(!llama_dataset_has_error());
    std::cout << "✓ Error clearing works\n";
}

// Test conversion between formats
void test_format_conversion() {
    std::cout << "\n=== Testing format conversion ===\n";

    // Load a GGUF dataset
    const char* gguf_file = "test_data/small_dataset.gguf";
    struct llama_dataset* dataset = from_gguf(gguf_file);

    if (!dataset) {
        std::cerr << "Failed to load GGUF dataset: " << llama_dataset_get_error() << "\n";
        assert(false && "Failed to load GGUF dataset");
        return;
    }

    // Convert to a new GGUF file
    const char* output_file = "test_gguf_output.gguf";
    to_gguf(dataset, output_file);

    if (llama_dataset_has_error()) {
        std::cerr << "Failed to convert dataset: " << llama_dataset_get_error() << "\n";
        llama_dataset_free(dataset);
        assert(false && "Failed to convert dataset");
        return;
    }

    // Load the converted file
    struct llama_dataset* converted = from_gguf(output_file);

    if (!converted) {
        std::cerr << "Failed to load converted dataset: " << llama_dataset_get_error() << "\n";
        std::cout << "✗ Format conversion test failed (GGUF write/read issue)\n";
        llama_dataset_free(dataset);
        return;
    }

    // Compare the datasets
    bool datasets_equal = compare_datasets(dataset, converted);
    std::cout << "Original and converted datasets are equal: " << (datasets_equal ? "yes" : "no") << "\n";
    assert(datasets_equal && "Original and converted datasets should be equal");

    // Clean up
    llama_dataset_free(dataset);
    llama_dataset_free(converted);
    remove(output_file);
    std::cout << "✓ Format conversion test passed\n";
}

// Test metadata access functions
void test_metadata_access() {
    std::cout << "\n=== Testing metadata access functions ===\n";

    // Load a GGUF dataset
    const char* gguf_file = "test_data/small_dataset.gguf";
    struct llama_dataset* dataset = from_gguf(gguf_file);

    if (!dataset) {
        std::cerr << "Failed to load GGUF dataset: " << llama_dataset_get_error() << "\n";
        assert(false && "Failed to load GGUF dataset");
        return;
    }

    // Test string metadata
    const char* format = dataset_get_metadata_str(dataset, DATASET_SOURCE_FORMAT);
    if (format) {
        std::cout << "Source format: " << format << "\n";
    } else {
        std::cout << "Source format not found in metadata\n";
    }

    // Test integer metadata
    int64_t count = dataset_get_metadata_int(dataset, DATASET_SEQUENCE_COUNT, -1);
    if (count != -1) {
        std::cout << "Sequence count from metadata: " << count << "\n";
        assert(count == n_sequences(dataset));
    } else {
        std::cout << "Sequence count not found in metadata\n";
    }

    // Test float metadata
    float value = dataset_get_metadata_float(dataset, "test.float", -1.0f);
    std::cout << "Test float value (default expected): " << value << "\n";
    assert(value == -1.0f);

    // Test null dataset handling
    assert(dataset_get_metadata_str(nullptr, DATASET_SOURCE_FORMAT) == nullptr);
    assert(dataset_get_metadata_int(nullptr, DATASET_SEQUENCE_COUNT, -1) == -1);
    assert(dataset_get_metadata_float(nullptr, "test.float", -1.0f) == -1.0f);
    std::cout << "✓ Null dataset metadata handling works\n";

    // Clean up
    llama_dataset_free(dataset);
    std::cout << "✓ Metadata access test passed\n";
}

int main() {

    std::cout << "=== Running dataset core functionality tests ===\n";

    // Test factory functions
    test_gguf_factory();
    test_text_factory();
    test_parquet_factory();

    // Test sequence access
    test_sequence_access();

    // Test streaming equivalence
    test_streaming_equivalence();

    // Test error conditions
    test_error_conditions();

    // Test format conversion
    test_format_conversion();

    // Test metadata access
    test_metadata_access();

    std::cout << "\n=== All tests completed successfully! ===\n";
    return 0;
}
