#include "test_core_functionality.h"

// Test streaming vs full loading equivalence
void test_streaming_equivalence_detailed();
void test_streaming_equivalence_detailed() {
    TEST_LOG_SECTION("Testing streaming vs full loading equivalence");

    // Check if streaming is supported
    bool supports_streaming = llama_dataset_supports_streaming(DATASET_GGUF, TEST_DATA_SMALL_GGUF);
    TEST_LOG_INFO("GGUF streaming supported: %s", supports_streaming ? "yes" : "no");

    // Test streaming equivalence using shared utility
    TEST_ASSERT(test_streaming_equivalence(TEST_DATA_SMALL_GGUF, DATASET_GGUF), 
                "Streaming and non-streaming should produce identical results");

    TEST_LOG_SUCCESS("Streaming equivalence test passed");
}

int main() {
    LLAMA_LOG_INFO("=== Running streaming functionality tests ===\n");

    // Test streaming equivalence
    test_streaming_equivalence_detailed();

    LLAMA_LOG_INFO("\n=== All streaming functionality tests completed successfully! ===\n");
    return 0;
}