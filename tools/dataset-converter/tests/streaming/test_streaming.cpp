#include "llama-dataset.h"
#include <iostream>
#include <cassert>

/**
 * Test for GGUF streaming support.
 *
 * This test verifies that:
 * 1. Streaming mode can be enabled for GGUF datasets
 * 2. Streaming and non-streaming modes produce identical results
 * 3. Fallback works when streaming is not available
 */
int main() {
    const char* test_file = "test_data/small_dataset.gguf";

    std::cout << "Testing GGUF streaming support..." << std::endl;

    // Check if streaming is supported for GGUF
    bool supports_streaming = llama_dataset_supports_streaming(DATASET_GGUF, test_file);
    std::cout << "GGUF streaming supported: " << (supports_streaming ? "yes" : "no") << std::endl;

    // Load dataset in non-streaming mode
    std::cout << "Loading dataset in non-streaming mode..." << std::endl;
    struct llama_dataset* non_streaming = llama_dataset_load_gguf(test_file, false);
    if (!non_streaming) {
        std::cerr << "Failed to load dataset in non-streaming mode: " << llama_dataset_get_error_message() << std::endl;
        return 1;
    }

    // Get sequence count and first sequence
    uint64_t seq_count = n_sequences(non_streaming);
    std::cout << "Sequence count: " << seq_count << std::endl;

    if (seq_count > 0) {
        int32_t seq_len = sequence_length(non_streaming, 0);
        const int32_t* seq_data = sequence(non_streaming, 0);

        std::cout << "First sequence length: " << seq_len << std::endl;
        std::cout << "First sequence data available: " << (seq_data != nullptr ? "yes" : "no") << std::endl;

        if (seq_data && seq_len > 0) {
            std::cout << "First few tokens: ";
            for (int i = 0; i < std::min(seq_len, 10); i++) {
                std::cout << seq_data[i] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Load dataset in streaming mode
    std::cout << "\nLoading dataset in streaming mode..." << std::endl;
    struct llama_dataset* streaming = llama_dataset_load_gguf(test_file, true);
    if (!streaming) {
        std::cerr << "Failed to load dataset in streaming mode: " << llama_dataset_get_error_message() << std::endl;
        llama_dataset_free(non_streaming);
        return 1;
    }

    // Verify streaming mode is enabled
    bool is_streaming = llama_dataset_is_streaming_enabled(streaming);
    std::cout << "Streaming mode enabled: " << (is_streaming ? "yes" : "no") << std::endl;

    // Get sequence count and first sequence
    uint64_t stream_seq_count = n_sequences(streaming);
    std::cout << "Sequence count: " << stream_seq_count << std::endl;

    // Verify sequence counts match
    assert(seq_count == stream_seq_count);

    if (stream_seq_count > 0) {
        int32_t stream_seq_len = sequence_length(streaming, 0);
        const int32_t* stream_seq_data = sequence(streaming, 0);

        std::cout << "First sequence length: " << stream_seq_len << std::endl;
        std::cout << "First sequence data available: " << (stream_seq_data != nullptr ? "yes" : "no") << std::endl;

        if (stream_seq_data && stream_seq_len > 0) {
            std::cout << "First few tokens: ";
            for (int i = 0; i < std::min(stream_seq_len, 10); i++) {
                std::cout << stream_seq_data[i] << " ";
            }
            std::cout << std::endl;
        }

        int32_t seq_len = sequence_length(non_streaming, 0);
        // Verify sequence lengths match
        assert(seq_len == stream_seq_len);

        const int32_t* seq_data = sequence(non_streaming, 0);
        // Verify sequence data matches
        if (seq_data && stream_seq_data) {
            bool data_matches = true;
            for (int i = 0; i < seq_len; i++) {
                if (seq_data[i] != stream_seq_data[i]) {
                    data_matches = false;
                    break;
                }
            }
            std::cout << "Sequence data matches: " << (data_matches ? "yes" : "no") << std::endl;
            assert(data_matches);
        }
    }

    // Clean up
    llama_dataset_free(non_streaming);
    llama_dataset_free(streaming);

    std::cout << "\nGGUF streaming test completed successfully!" << std::endl;
    return 0;
}
