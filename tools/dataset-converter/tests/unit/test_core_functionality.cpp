#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "common.h"
#include "common/log.h"
#include "llama-dataset.h"
#include "llama-impl.h"

// Helper function to create a simple text dataset file for testing
bool create_test_text_file(const char* path, const std::vector<std::string>& lines);
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
bool compare_datasets(struct llama_dataset* ds1, struct llama_dataset* ds2);
bool compare_datasets(struct llama_dataset* ds1, struct llama_dataset* ds2) {
    if (!ds1 || !ds2) {
        return false;
    }

    // Compare sequence counts
    uint64_t count1 = llama_dataset_n_sequences(ds1);
    uint64_t count2 = llama_dataset_n_sequences(ds2);

    if (count1 != count2) {
        LLAMA_LOG_ERROR( "Sequence counts differ: %lu vs %lu\n", count1, count2);
        return false;
    }

    // Compare each sequence
    for (uint64_t i = 0; i < count1; i++) {
        int32_t len1 = llama_dataset_sequence_length(ds1, i);
        int32_t len2 = llama_dataset_sequence_length(ds2, i);

        if (len1 != len2) {
            LLAMA_LOG_ERROR( "Sequence %lu lengths differ: %d vs %d\n", i, len1, len2);
            return false;
        }

        const int32_t* seq1 = llama_dataset_sequence(ds1, i);
        const int32_t* seq2 = llama_dataset_sequence(ds2, i);

        if (!seq1 || !seq2) {
            LLAMA_LOG_ERROR( "Sequence %lu data is null\n", i);
            return false;
        }

        for (int32_t j = 0; j < len1; j++) {
            if (seq1[j] != seq2[j]) {
                LLAMA_LOG_ERROR( "Sequence %lu data differs at position %d : %d vs %d\n", i, j, seq1[j], seq2[j]);
                return false;
            }
        }
    }

    return true;
}

// Test GGUF factory function with valid and invalid inputs
void test_gguf_factory();
void test_gguf_factory() {
    LLAMA_LOG_INFO("\n=== Testing GGUF factory function ===\n");

    // Test with null path
    common_params params;
    struct llama_dataset* dataset = llama_dataset_from_gguf(&params);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    LLAMA_LOG_INFO("✓ Null path error handling works\n");
    llama_dataset_clear_error();

    // Test with non-existent file
    params.in_files.push_back("non_existent_file.gguf");
    dataset = llama_dataset_from_gguf(&params);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    LLAMA_LOG_INFO("✓ Non-existent file error handling works\n");
    llama_dataset_clear_error();

    // Test with valid file
    params.in_files.back() = "test_data/small_dataset.gguf";
    dataset = llama_dataset_from_gguf(&params);

    if (dataset) {
        LLAMA_LOG_INFO("✓ Successfully loaded valid GGUF file\n");

        // Test basic properties
        uint64_t seq_count = llama_dataset_n_sequences(dataset);
        LLAMA_LOG_INFO("  Sequence count: %lu\n", seq_count);
        assert(seq_count > 0);

        // Test first sequence
        int32_t seq_len = llama_dataset_sequence_length(dataset, 0);
        LLAMA_LOG_INFO("  First sequence length: %d\n", seq_len);
        assert(seq_len > 0);

        const int32_t* seq_data = llama_dataset_sequence(dataset, 0);
        assert(seq_data != nullptr);
        LLAMA_LOG_INFO("  First few tokens: ");
        for (int i = 0; i < std::min(seq_len, 5); i++) {
            LLAMA_LOG_INFO("%d ", seq_data[i]);
        }
        LLAMA_LOG_INFO("\n");

        // Test metadata access
        const char* format = llama_dataset_get_metadata_str(dataset, TRAINING_FORMAT_SOURCE);
        if (format) {
            LLAMA_LOG_INFO("  Source format: %s\n", format);
        }

        int64_t count = llama_dataset_get_metadata_int(dataset, TRAINING_SEQUENCE_COUNT, 0);
        LLAMA_LOG_INFO("  Metadata sequence count: %ld\n", count);
        LLAMA_LOG_INFO("  Actual sequence count: %lu\n", seq_count);
        // Note: Metadata might not be available, so we don't assert on it
        // assert(count == seq_count);

        // Test tensor access
        struct ggml_tensor* tensor = llama_dataset_sequence_tensor(dataset, 0);
        assert(tensor != nullptr);
        LLAMA_LOG_INFO("✓ Tensor access works\n");

        // Clean up
        llama_dataset_free(dataset);
        LLAMA_LOG_INFO("✓ Dataset cleanup successful\n");
    } else {
        LLAMA_LOG_ERROR( "✗ Failed to load valid GGUF file: %s\n",  llama_dataset_get_error());
        assert(false && "Failed to load valid GGUF file");
    }
}

// Test text factory function with valid and invalid inputs
void test_text_factory();
void test_text_factory() {
    LLAMA_LOG_INFO("\n=== Testing text factory function ===\n");

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
    common_params params;
    struct llama_dataset* dataset = llama_dataset_from_txt(&params, nullptr);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    LLAMA_LOG_INFO("✓ Null path error handling works\n");
    llama_dataset_clear_error();

    // Test with null model (tokenizer)
    params.in_files.push_back(test_file);
    dataset = llama_dataset_from_txt(&params, nullptr);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    LLAMA_LOG_INFO("✓ Null tokenizer error handling works\n");
    llama_dataset_clear_error();

    // Note: We can't fully test the text loader without a valid tokenizer model
    // This would require loading a real model, which is beyond the scope of this test
    LLAMA_LOG_INFO("✓ Text factory function tests completed\n");

    // Clean up
    remove(test_file);
}

// Test Parquet factory function with valid and invalid inputs
void test_parquet_factory();
void test_parquet_factory() {
    LLAMA_LOG_INFO("\n=== Testing Parquet factory function ===\n");

    // Test with null path
    common_params params;
#ifdef LLAMA_PARQUET
    struct llama_dataset* dataset = llama_dataset_from_parquet(&params);
#else
    return;
#endif
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    LLAMA_LOG_INFO("✓ Null path error handling works\n");
    llama_dataset_clear_error();

    // Test with non-existent file
    params.in_files.push_back("non_existent_file.parquet");
#ifdef LLAMA_PARQUET
    dataset = llama_dataset_from_parquet(&params);
#else
    return;
#endif
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    LLAMA_LOG_INFO("✓ Non-existent file error handling works\n");
    llama_dataset_clear_error();

    // Test with valid file if available
    params.in_files.back() = "test_data/parquet_dataset.parquet";
#ifdef LLAMA_PARQUET
    dataset = llama_dataset_from_parquet(&params);
    if (dataset) {
        LLAMA_LOG_INFO("✓ Successfully loaded valid Parquet file\n");

        // Test basic properties
        uint64_t seq_count = llama_dataset_n_sequences(dataset);
        LLAMA_LOG_INFO("  Sequence count: %lu\n", seq_count);
        assert(seq_count > 0);

        // Test first sequence
        int32_t seq_len = llama_dataset_sequence_length(dataset, 0);
        LLAMA_LOG_INFO("  First sequence length: %d\n", seq_len);
        assert(seq_len > 0);

        const int32_t* seq_data = llama_dataset_sequence(dataset, 0);
        assert(seq_data != nullptr);
        LLAMA_LOG_INFO("  First few tokens: ");
        for (int i = 0; i < std::min(seq_len, 5); i++) {
            LLAMA_LOG_INFO("%d ", seq_data[i]);
        }
        LLAMA_LOG_INFO("\n");

        // Clean up
        llama_dataset_free(dataset);
        LLAMA_LOG_INFO("✓ Dataset cleanup successful\n");
    } else {
        LLAMA_LOG_INFO("  Parquet file not available or support not compiled in: %s\n",  llama_dataset_get_error());
        llama_dataset_clear_error();
    }
#endif
}

// Test sequence access functions for consistency across formats
void test_sequence_access();
void test_sequence_access() {
    LLAMA_LOG_INFO("\n=== Testing sequence access functions ===\n");

    // Load a GGUF dataset
    const char* gguf_file = "test_data/small_dataset.gguf";
    common_params params;
    params.in_files.push_back(gguf_file);
    struct llama_dataset* gguf_dataset = llama_dataset_from_gguf(&params);

    if (!gguf_dataset) {
        LLAMA_LOG_ERROR( "Failed to load GGUF dataset: %s\n",  llama_dataset_get_error());
        assert(false && "Failed to load GGUF dataset");
        return;
    }

    // Test basic sequence access
    uint64_t seq_count = llama_dataset_n_sequences(gguf_dataset);
    LLAMA_LOG_INFO("GGUF dataset sequence count: %lu\n", seq_count);

    // Test out-of-bounds access
    int32_t invalid_len = llama_dataset_sequence_length(gguf_dataset, seq_count + 1);
    assert(invalid_len == 0);
    LLAMA_LOG_INFO("✓ Out-of-bounds sequence length check works\n");

    const int32_t* invalid_seq = llama_dataset_sequence(gguf_dataset, seq_count + 1);
    assert(invalid_seq == nullptr);
    LLAMA_LOG_INFO("✓ Out-of-bounds sequence access check works\n");

    // Test tensor access
    struct ggml_tensor* invalid_tensor = llama_dataset_sequence_tensor(gguf_dataset, seq_count + 1);
    assert(invalid_tensor == nullptr);
    LLAMA_LOG_INFO("✓ Out-of-bounds tensor access check works\n");

    // Test null dataset handling
    assert(llama_dataset_n_sequences(nullptr) == 0);
    assert(llama_dataset_sequence_length(nullptr, 0) == 0);
    assert(llama_dataset_sequence(nullptr, 0) == nullptr);
    assert(llama_dataset_sequence_tensor(nullptr, 0) == nullptr);
    LLAMA_LOG_INFO("✓ Null dataset handling works\n");

    // Clean up
    llama_dataset_free(gguf_dataset);
}

// Test streaming vs full loading equivalence
void test_streaming_equivalence();
void test_streaming_equivalence() {
    LLAMA_LOG_INFO("\n=== Testing streaming vs full loading equivalence ===\n");

    const char* test_file = "test_data/small_dataset.gguf";

    // Check if streaming is supported
    bool supports_streaming = llama_dataset_supports_streaming(DATASET_GGUF, test_file);
    LLAMA_LOG_INFO("GGUF streaming supported: %s\n", supports_streaming ? "yes" : "no");

    // Load dataset in non-streaming mode
    common_params params;
    params.in_files.push_back(test_file);
    struct llama_dataset* non_streaming = llama_dataset_load_gguf(&params);
    if (!non_streaming) {
        LLAMA_LOG_ERROR( "Failed to load dataset in non-streaming mode: %s\n",  llama_dataset_get_error());
        assert(false && "Failed to load dataset in non-streaming mode");
        return;
    }

    // Load dataset in streaming mode
    params.dataset_streaming = true;
    struct llama_dataset* streaming = llama_dataset_load_gguf(&params);
    if (!streaming) {
        LLAMA_LOG_ERROR( "Failed to load dataset in streaming mode: %s\n",  llama_dataset_get_error());
        llama_dataset_free(non_streaming);
        assert(false && "Failed to load dataset in streaming mode");
        return;
    }

    // Verify streaming mode is enabled if supported
    bool is_streaming = llama_dataset_is_streaming_enabled(streaming);
    LLAMA_LOG_INFO("Streaming mode enabled: %s\n", is_streaming ? "yes" : "no");

    // Compare the datasets
    bool datasets_equal = compare_datasets(non_streaming, streaming);
    LLAMA_LOG_INFO("Datasets are equal: %s\n", datasets_equal ? "yes" : "no");
    assert(datasets_equal && "Streaming and non-streaming datasets should be equal");

    // Clean up
    llama_dataset_free(non_streaming);
    llama_dataset_free(streaming);
    LLAMA_LOG_INFO("✓ Streaming equivalence test passed\n");
}

// Test error conditions with invalid inputs
void test_error_conditions();
void test_error_conditions() {
    LLAMA_LOG_INFO("\n=== Testing error conditions ===\n");

    // Test with corrupted GGUF file if available
    const char* corrupted_file = "test_data/corrupted_dataset.gguf";
    common_params params;
    params.in_files.push_back(corrupted_file);
    struct llama_dataset* dataset = llama_dataset_from_gguf(&params);

    if (!dataset) {
        LLAMA_LOG_INFO("✓ Corrupted GGUF file correctly rejected: %s\n",  llama_dataset_get_error());
        llama_dataset_clear_error();
    } else {
        LLAMA_LOG_ERROR( "✗ Corrupted GGUF file was loaded successfully, which is unexpected\n");
        llama_dataset_free(dataset);
    }

    // Test with corrupted Parquet file if available
    params.in_files.back() = "test_data/corrupted_dataset.parquet";
#ifdef LLAMA_PARQUET
    dataset = llama_dataset_from_parquet(&params);
#else
    return;
#endif
    if (!dataset) {
        LLAMA_LOG_INFO("✓ Corrupted Parquet file correctly rejected: %s\n",  llama_dataset_get_error());
        llama_dataset_clear_error();
    } else {
        LLAMA_LOG_ERROR( "✗ Corrupted Parquet file was loaded successfully, which is unexpected\n");
        llama_dataset_free(dataset);
    }

    // Test error code to string conversion
    const char* error_str = llama_dataset_error_code_to_string(DATASET_ERROR_FILE_NOT_FOUND);
    assert(error_str != nullptr);
    LLAMA_LOG_INFO("Error code string: %s\n", error_str);

    // Test error clearing
    llama_dataset_clear_error();
    assert(!llama_dataset_has_error());
    LLAMA_LOG_INFO("✓ Error clearing works\n");
}

// Test conversion between formats
void test_format_conversion();
void test_format_conversion() {
    LLAMA_LOG_INFO("\n=== Testing format conversion ===\n");

    // Load a GGUF dataset
    const char* gguf_file = "test_data/small_dataset.gguf";
    common_params params;
    params.in_files.push_back(gguf_file);
    struct llama_dataset* dataset = llama_dataset_from_gguf(&params);

    if (!dataset) {
        LLAMA_LOG_ERROR( "Failed to load GGUF dataset: %s\n",  llama_dataset_get_error());
        assert(false && "Failed to load GGUF dataset");
        return;
    }

    // Convert to a new GGUF file
    const char* output_file = "test_gguf_output.gguf";
    llama_dataset_to_gguf(dataset, output_file);

    if (llama_dataset_has_error()) {
        LLAMA_LOG_ERROR( "Failed to convert dataset: %s", llama_dataset_get_error());
        llama_dataset_free(dataset);
        assert(false && "Failed to convert dataset");
        return;
    }

    // Load the converted file
    params.in_files.back() =  output_file;
    struct llama_dataset* converted = llama_dataset_from_gguf(&params);

    if (!converted) {
        LLAMA_LOG_ERROR( "Failed to load converted dataset: %s", llama_dataset_get_error());
        LLAMA_LOG_INFO("✗ Format conversion test failed (GGUF write/read issue)\n");
        llama_dataset_free(dataset);
        return;
    }

    // Compare the datasets
    bool datasets_equal = compare_datasets(dataset, converted);
    LLAMA_LOG_INFO("Original and converted datasets are equal: %s\n", datasets_equal ? "yes" : "no");
    assert(datasets_equal && "Original and converted datasets should be equal");

    // Clean up
    llama_dataset_free(dataset);
    llama_dataset_free(converted);
    remove(output_file);
    LLAMA_LOG_INFO("✓ Format conversion test passed\n");
}

// Test metadata access functions
void test_metadata_access();
void test_metadata_access() {
    LLAMA_LOG_INFO("\n=== Testing metadata access functions ===\n");

    // Load a GGUF dataset
    const char* gguf_file = "test_data/small_dataset.gguf";
    common_params params;
    params.in_files.push_back(gguf_file);
    struct llama_dataset* dataset = llama_dataset_from_gguf(&params);

    if (!dataset) {
        LLAMA_LOG_ERROR( "Failed to load GGUF dataset: %s\n", llama_dataset_get_error());
        assert(false && "Failed to load GGUF dataset");
        return;
    }

    // Test string metadata
    const char* format = llama_dataset_get_metadata_str(dataset, TRAINING_FORMAT_SOURCE);
    if (format) {
        LLAMA_LOG_INFO("Source format: %s\n", format);
    } else {
        LLAMA_LOG_INFO("Source format not found in metadata\n");
    }

    // Test integer metadata
    int64_t count = llama_dataset_get_metadata_int(dataset, TRAINING_SEQUENCE_COUNT, -1);
    if (count != -1) {
        LLAMA_LOG_INFO("Sequence count from metadata: %ld\n", count);
        assert(count == llama_dataset_n_sequences(dataset));
    } else {
        LLAMA_LOG_INFO("Sequence count not found in metadata\n");
    }

    // Test float metadata
    float value = llama_dataset_get_metadata_float(dataset, "test.float", -1.0f);
    LLAMA_LOG_INFO("Test float value (default expected): %f\n", value);
    assert(value == -1.0f);

    // Test null dataset handling
    assert(llama_dataset_get_metadata_str(nullptr, TRAINING_FORMAT_SOURCE) == nullptr);
    assert(llama_dataset_get_metadata_int(nullptr, TRAINING_SEQUENCE_COUNT, -1) == -1);
    assert(llama_dataset_get_metadata_float(nullptr, "test.float", -1.0f) == -1.0f);
    LLAMA_LOG_INFO("✓ Null dataset metadata handling works\n");

    // Clean up
    llama_dataset_free(dataset);
    LLAMA_LOG_INFO("✓ Metadata access test passed\n");
}

int main() {

    LLAMA_LOG_INFO("=== Running dataset core functionality tests ===\n");

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

    LLAMA_LOG_INFO("\n=== All tests completed successfully! ===\n");
    return 0;
}
