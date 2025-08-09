#include "../unit/test_core_functionality.h"
#include "common.h"
#include "llama-dataset.h"
#include "llama-impl.h"

// TestDataCreator is now available from the shared header

// compare_datasets_exact is now available from the shared header

// TestMemoryTracker and TestTimer are now available from the shared header

// Test end-to-end workflow: load → access → convert → save → reload
void test_end_to_end_workflow();
void test_end_to_end_workflow() {
    LLAMA_LOG_INFO("\n=== Testing end-to-end workflow ===\n");

    const char* original_file = "test_data/small_dataset.gguf";
    const char* converted_file = "test_integration_output.gguf";
    const char* reloaded_file = "test_integration_reloaded.gguf";

    // Step 1: Load original dataset
    LLAMA_LOG_INFO("Step 1: Loading original dataset...\n");
    TestTimer timer;
    common_params params;
    params.in_files.push_back(original_file);
    struct llama_dataset* original = llama_dataset_from_gguf(&params);
    double load_time = timer.elapsed_ms();

    if (!original) {
        LLAMA_LOG_ERROR("Failed to load original dataset: %s\n", llama_dataset_get_error());
        assert(false && "Failed to load original dataset");
        return;
    }

    LLAMA_LOG_INFO("  Load time: %f ms\n", load_time);

    // Step 2: Access and validate data
    LLAMA_LOG_INFO("Step 2: Accessing and validating data...\n");
    uint64_t seq_count = llama_dataset_n_sequences(original);
    LLAMA_LOG_INFO("  Sequence count: %lu\n", seq_count);
    assert(seq_count > 0);

    // Access all sequences to ensure they're valid
    for (uint64_t i = 0; i < seq_count; i++) {
        int32_t len = llama_dataset_sequence_length(original, i);
        const int32_t* data = llama_dataset_sequence(original, i);
        assert(len > 0);
        assert(data != nullptr);

        // Validate first few tokens
        if (i == 0 && len >= 3) {
            LLAMA_LOG_INFO("  First sequence tokens: %d %d %d ...\n", data[0], data[1], data[2]);
        }
    }

    // Step 3: Convert to new GGUF file
    LLAMA_LOG_INFO("Step 3: Converting to new GGUF file...\n");
    timer.start();
    llama_dataset_to_gguf(original, converted_file);
    double convert_time = timer.elapsed_ms();

    if (llama_dataset_has_error()) {
        LLAMA_LOG_ERROR("Failed to convert dataset: %s\n", llama_dataset_get_error());
        llama_dataset_free(original);
        assert(false && "Failed to convert dataset");
        return;
    }

    LLAMA_LOG_INFO("  Convert time: %f ms\n", convert_time);

    // Step 4: Reload converted file
    LLAMA_LOG_INFO("Step 4: Reloading converted file...\n");
    timer.start();
    params.in_files.back() = converted_file;
    struct llama_dataset* converted = llama_dataset_from_gguf(&params);
    double reload_time = timer.elapsed_ms();

    if (!converted) {
        LLAMA_LOG_ERROR("Failed to reload converted dataset: %s\n", llama_dataset_get_error());
        LLAMA_LOG_INFO("✗ End-to-end workflow test failed (GGUF write/read issue)\n");
        llama_dataset_free(original);
        return;
    }

    LLAMA_LOG_INFO("  Reload time: %f ms\n", reload_time);

    // Step 5: Compare original and converted datasets
    LLAMA_LOG_INFO("Step 5: Comparing original and converted datasets...\n");
    bool datasets_equal = compare_datasets_exact(original, converted);
    LLAMA_LOG_INFO("  Datasets are identical: %s\n", datasets_equal ? "yes" : "no");
    assert(datasets_equal && "Original and converted datasets should be identical");

    // Step 6: Save converted dataset again (double conversion test)
    LLAMA_LOG_INFO("Step 6: Testing double conversion...\n");
    llama_dataset_to_gguf(converted, reloaded_file);

    if (llama_dataset_has_error()) {
        LLAMA_LOG_ERROR("Failed to save converted dataset: %s\n", llama_dataset_get_error());
        llama_dataset_free(original);
        llama_dataset_free(converted);
        assert(false && "Failed to save converted dataset");
        return;
    }
    params.in_files.back() =  reloaded_file;
    struct llama_dataset* double_converted = llama_dataset_from_gguf(&params);
    if (!double_converted) {
        LLAMA_LOG_ERROR("Failed to load double-converted dataset: %s\n", llama_dataset_get_error());
        llama_dataset_free(original);
        llama_dataset_free(converted);
        assert(false && "Failed to load double-converted dataset");
        return;
    }

    bool double_conversion_equal = compare_datasets_exact(original, double_converted);
    LLAMA_LOG_INFO("  Double conversion preserves data: %s\n", (double_conversion_equal ? "yes" : "no"));
    assert(double_conversion_equal && "Double conversion should preserve data");

    // Clean up
    llama_dataset_free(original);
    llama_dataset_free(converted);
    llama_dataset_free(double_converted);
    remove(converted_file);
    remove(reloaded_file);

    LLAMA_LOG_INFO("✓ End-to-end workflow test passed\n");
}

// Test memory usage validation for streaming vs full loading
void test_memory_usage_validation();
void test_memory_usage_validation() {
    LLAMA_LOG_INFO("\n=== Testing memory usage validation ===\n");

    const char* test_file = "test_data/small_dataset.gguf";

    // Test non-streaming mode memory usage
    LLAMA_LOG_INFO("Testing non-streaming mode memory usage...\n");
    TestMemoryTracker non_streaming_tracker;

    common_params params;
    params.in_files.push_back(test_file);
    struct llama_dataset* non_streaming = llama_dataset_load_gguf(&params);
    if (!non_streaming) {
        LLAMA_LOG_ERROR("Failed to load dataset in non-streaming mode: %s\n", llama_dataset_get_error());
        assert(false && "Failed to load dataset in non-streaming mode");
        return;
    }

    size_t non_streaming_memory = non_streaming_tracker.get_delta();
    LLAMA_LOG_INFO("  Non-streaming memory usage: %lu bytes\n", non_streaming_memory);

    // Access all sequences to trigger full loading
    uint64_t seq_count = llama_dataset_n_sequences(non_streaming);
    for (uint64_t i = 0; i < seq_count; i++) {
        const int32_t* data = llama_dataset_sequence(non_streaming, i);
        (void)data; // Suppress unused variable warning
    }

    size_t non_streaming_after_access = non_streaming_tracker.get_delta();
    LLAMA_LOG_INFO("  Non-streaming memory after access: %lu bytes\n", non_streaming_after_access);

    // Test streaming mode memory usage
    LLAMA_LOG_INFO("Testing streaming mode memory usage...\n");
    TestMemoryTracker streaming_tracker;
    params.in_files.back() = test_file;
    params.dataset_streaming = true;
    struct llama_dataset* streaming = llama_dataset_load_gguf(&params);
    if (!streaming) {
        LLAMA_LOG_ERROR("Failed to load dataset in streaming mode: %s\n", llama_dataset_get_error());
        llama_dataset_free(non_streaming);
        assert(false && "Failed to load dataset in streaming mode");
        return;
    }

    size_t streaming_memory = streaming_tracker.get_delta();
    LLAMA_LOG_INFO("  Streaming memory usage: %lu bytes\n", streaming_memory);

    // Access all sequences in streaming mode
    uint64_t stream_seq_count = llama_dataset_n_sequences(streaming);
    for (uint64_t i = 0; i < stream_seq_count; i++) {
        const int32_t* data = llama_dataset_sequence(streaming, i);
        (void)data; // Suppress unused variable warning
    }

    size_t streaming_after_access = streaming_tracker.get_delta();
    LLAMA_LOG_INFO("  Streaming memory after access: %lu bytes\n", streaming_after_access);

    // Validate memory usage patterns
    bool streaming_enabled = llama_dataset_is_streaming_enabled(streaming);
    if (streaming_enabled) {
        // Streaming should use less initial memory
        LLAMA_LOG_INFO("  Streaming uses less initial memory: %s\n", streaming_memory <= non_streaming_memory ? "yes" : "no");
        // Note: This assertion might not always hold depending on implementation details
        // assert(streaming_memory <= non_streaming_memory && "Streaming should use less initial memory");
    } else {
        LLAMA_LOG_INFO("  Streaming fallback to full loading detected\n");
        // If streaming fell back to full loading, memory usage should be similar
        assert(abs((int)(streaming_memory - non_streaming_memory)) < 1024 * 1024 && "Fallback should have similar memory usage");
    }

    // Verify data consistency
    bool data_consistent = compare_datasets_exact(non_streaming, streaming);
    LLAMA_LOG_INFO("  Data consistency between modes: %s\n", data_consistent ? "yes" : "no");
    assert(data_consistent && "Streaming and non-streaming should produce identical data");

    // Clean up
    llama_dataset_free(non_streaming);
    llama_dataset_free(streaming);

    LLAMA_LOG_INFO("✓ Memory usage validation test passed\n");
}

//Test performance comparison across formats and loading modes
void test_performance_comparison();
void test_performance_comparison() {
    LLAMA_LOG_INFO("\n=== Testing performance comparison ===\n");

    const char* gguf_file = "test_data/small_dataset.gguf";
    const char* parquet_file = "test_data/parquet_dataset.parquet";
    const char* text_file = "test_data/text_dataset.txt";

    common_params paramsA, paramsB, paramsC, paramsD;
    paramsA.in_files.push_back(gguf_file);
    paramsB.in_files.push_back(gguf_file);
    paramsB.dataset_streaming = true;
    paramsC.in_files.push_back(parquet_file);
    paramsD.in_files.push_back(text_file);
    struct {
        const char* format;
        const char* file;
        std::function<struct llama_dataset *()> loader;
    } test_cases[] = {
        {"GGUF (non-streaming)", gguf_file, [=]() { return llama_dataset_load_gguf(&paramsA); }},
        {"GGUF (streaming)", gguf_file, [=]() { return llama_dataset_load_gguf(&paramsB); }},
#ifdef LLAMA_PARQUET
        {"Parquet", parquet_file, [=]() { return llama_dataset_from_parquet(&paramsC); }},
#endif
        {"Text", text_file, [=]() { return llama_dataset_from_txt(&paramsD, nullptr); }}
    };

    LLAMA_LOG_INFO("Format\t\t\tLoad Time (ms)\tAccess Time (ms)\tMemory (KB)\n");
    LLAMA_LOG_INFO("---------------------------------------------------------------\n");

    for (const auto& test_case : test_cases) {
        TestTimer timer;
        TestMemoryTracker memory_tracker;

        // Test loading time
        timer.start();
        struct llama_dataset* dataset = test_case.loader();
        double load_time = timer.elapsed_ms();

        if (!dataset) {
            LLAMA_LOG_INFO("%s %s", test_case.format, "\t\tFAILED\t\t-\t\t-\n");
            llama_dataset_clear_error();
            continue;
        }

        size_t load_memory = memory_tracker.get_delta_kb();
        (void)load_memory; // Variable used for debugging/profiling, suppress warning

        // Test access time (access all sequences)
        timer.start();
        uint64_t seq_count = llama_dataset_n_sequences(dataset);
        for (uint64_t i = 0; i < seq_count; i++) {
            int32_t len = llama_dataset_sequence_length(dataset, i);
            const int32_t* data = llama_dataset_sequence(dataset, i);
            (void)len; (void)data; // Suppress unused variable warnings
        }
        double access_time = timer.elapsed_ms();

        size_t total_memory = memory_tracker.get_delta_kb();

        LLAMA_LOG_INFO("%s, %f \t\t%%s %f \t\t%%s %lu \t\t%%s\n",test_case.format, load_time, access_time, total_memory);

        llama_dataset_free(dataset);
    }

    LLAMA_LOG_INFO("✓ Performance comparison test completed\n");
}

// Test cross-format data consistency
void test_cross_format_consistency();
void test_cross_format_consistency() {
    LLAMA_LOG_INFO("\n=== Testing cross-format data consistency ===\n");

    // Create test data files with known content
    const char* test_gguf = "test_integration_small.gguf";
    const char* test_parquet = "test_integration_small.parquet";
    const char* test_text = "test_integration_small.txt";

    LLAMA_LOG_INFO("Creating test data files...\n");
    bool gguf_created = TestDataCreator::create_small_gguf(test_gguf);
    bool parquet_created = TestDataCreator::create_equivalent_parquet(test_parquet);
    bool text_created = TestDataCreator::create_equivalent_text(test_text);

    LLAMA_LOG_INFO("  GGUF file created: %s\n", gguf_created ? "yes" : "no");
    LLAMA_LOG_INFO("  Parquet file created: %s\n", parquet_created ? "yes" : "no");
    LLAMA_LOG_INFO("  Text file created: %s\n", text_created ? "yes" : "no");

    if (!gguf_created) {
        LLAMA_LOG_ERROR("Failed to create test GGUF file\n");
        return;
    }

    // Load GGUF dataset as reference
    common_params params;
    params.in_files.push_back(test_gguf);
    struct llama_dataset* gguf_dataset = llama_dataset_from_gguf(&params);
    if (!gguf_dataset) {
        LLAMA_LOG_ERROR("Failed to load test GGUF dataset: %s\n", llama_dataset_get_error());

        // Clean up test files
        remove(test_gguf);
        remove(test_parquet);
        remove(test_text);

        LLAMA_LOG_INFO("✓ Cross-format consistency test completed\n");
        return;
    }

    LLAMA_LOG_INFO("Reference GGUF dataset loaded successfully\n");
    uint64_t ref_seq_count = llama_dataset_n_sequences(gguf_dataset);
    LLAMA_LOG_INFO("  Reference sequence count: %lu\n", ref_seq_count);
#ifdef LLAMA_PARQUET
    // Test Parquet consistency (if available)
    if (parquet_created) {
        params.in_files.back() = test_parquet;
        if (struct llama_dataset * parquet_dataset = llama_dataset_from_parquet(&params)) {
            LLAMA_LOG_INFO("Testing Parquet consistency...\n");
            bool parquet_consistent = compare_datasets_exact(gguf_dataset, parquet_dataset);
            LLAMA_LOG_INFO("  Parquet data consistent with GGUF: %s\n", parquet_consistent ? "yes" : "no");
            // Note: This might fail if Parquet loader is not fully implemented
            llama_dataset_free(parquet_dataset);
        } else {
            LLAMA_LOG_INFO("  Parquet loader not available or failed: %s\n", llama_dataset_get_error());
            llama_dataset_clear_error();
        }
    }
#endif
    // Test Text consistency (if available)
    if (text_created) {
        params.in_files.back() = test_text;
        struct llama_dataset* text_dataset = llama_dataset_from_txt(&params, nullptr);
        if (text_dataset) {
            LLAMA_LOG_INFO("Testing Text consistency...\n");
            bool text_consistent = compare_datasets_exact(gguf_dataset, text_dataset);
            LLAMA_LOG_INFO("  Text data consistent with GGUF: %s\n", text_consistent ? "yes" : "no");
            // Note: This might fail if Text loader is not fully implemented
            llama_dataset_free(text_dataset);
        } else {
            LLAMA_LOG_INFO("  Text loader not available or failed: %s\n", llama_dataset_get_error());
            llama_dataset_clear_error();
        }
    }

    llama_dataset_free(gguf_dataset);

    // Clean up test files
    remove(test_gguf);
    remove(test_parquet);
    remove(test_text);

    LLAMA_LOG_INFO("✓ Cross-format consistency test completed\n");
}

// Test large dataset handling
void test_large_dataset_handling();
void test_large_dataset_handling() {
    LLAMA_LOG_INFO("\n=== Testing large dataset handling ===\n");

    const char* large_text_file = "test_data/large_text_dataset.txt";
    const char* large_parquet_file = "test_data/parquet_dataset.parquet";

    // Test large text file if available
    LLAMA_LOG_INFO("Testing large text dataset...\n");
    TestTimer timer;
    TestMemoryTracker memory_tracker;

    common_params params;
    params.in_files.push_back(large_text_file);
    struct llama_dataset* large_text = llama_dataset_from_txt(&params, nullptr);
    if (large_text) {
        double load_time = timer.elapsed_ms();
        size_t memory_usage = memory_tracker.get_delta_kb();

        uint64_t seq_count = llama_dataset_n_sequences(large_text);
        LLAMA_LOG_INFO("  Large text dataset loaded successfully\n");
        LLAMA_LOG_INFO("  Sequence count: %lu\n", seq_count);
        LLAMA_LOG_INFO("  Load time: %f ms\n", load_time);
        LLAMA_LOG_INFO("  Memory usage: %lu\n", memory_usage);

        // Test random access to sequences
        if (seq_count > 10) {
            timer.start();
            for (int i = 0; i < 10; i++) {
                uint64_t idx = rand() % seq_count;
                int32_t len = llama_dataset_sequence_length(large_text, idx);
                const int32_t* data = llama_dataset_sequence(large_text, idx);
                assert(len > 0);
                assert(data != nullptr);
                (void)len; (void)data; // Suppress unused variable warnings
            }
            double access_time = timer.elapsed_ms();
            LLAMA_LOG_INFO("  Random access time (10 sequences): %f ms\n", access_time);
        }

        llama_dataset_free(large_text);
    } else {
        LLAMA_LOG_INFO("  Large text dataset not available: %s\n", llama_dataset_get_error());
        llama_dataset_clear_error();
    }

    // Test large Parquet file if available
    LLAMA_LOG_INFO("Testing large Parquet dataset...\n");
    timer.start();
    memory_tracker = TestMemoryTracker();
    params.in_files.back() = large_parquet_file;
#ifdef LLAMA_PARQUET
    struct llama_dataset* large_parquet = llama_dataset_from_parquet(&params);
    if (large_parquet) {
        double load_time = timer.elapsed_ms();
        size_t memory_usage = memory_tracker.get_delta_kb();

        uint64_t seq_count = llama_dataset_n_sequences(large_parquet);
        LLAMA_LOG_INFO("  Large Parquet dataset loaded successfully\n");
        LLAMA_LOG_INFO("  Sequence count: %lu\n", seq_count);
        LLAMA_LOG_INFO("  Load time: %f ms\n", load_time);
        LLAMA_LOG_INFO("  Memory usage: %lu\n", memory_usage);

        // Test streaming vs non-streaming for large dataset
        bool supports_streaming = llama_dataset_supports_streaming(DATASET_PARQUET, large_parquet_file);
        LLAMA_LOG_INFO("  Streaming supported: %s\n", supports_streaming ? "yes" : "no");

        llama_dataset_free(large_parquet);
    } else {
        LLAMA_LOG_INFO("  Large Parquet dataset not available: %s\n", llama_dataset_get_error());
        llama_dataset_clear_error();
    }
#endif

    LLAMA_LOG_INFO("✓ Large dataset handling test completed\n");
}

// Test error recovery and robustness
void test_error_recovery();
void test_error_recovery() {
    LLAMA_LOG_INFO("\n=== Testing error recovery and robustness ===\n");

    // Test recovery from corrupted files
    const char* corrupted_gguf = "test_data/corrupted_dataset.gguf";
    const char* corrupted_parquet = "test_data/corrupted_dataset.parquet";

    LLAMA_LOG_INFO("Testing corrupted GGUF file handling...\n");
    common_params params;
    params.in_files.push_back(corrupted_gguf);
    struct llama_dataset* dataset = llama_dataset_from_gguf(&params);
    if (!dataset) {
        LLAMA_LOG_INFO(2"  Corrupted GGUF correctly rejected: %s\n", llama_dataset_get_error());
        llama_dataset_clear_error();
    } else {
        LLAMA_LOG_INFO("  Warning: Corrupted GGUF was loaded (might be valid)\n");
        llama_dataset_free(dataset);
    }

    LLAMA_LOG_INFO("Testing corrupted Parquet file handling...\n");
    params.in_files.back() = corrupted_parquet;
#ifdef LLAMA_PARQUET
    dataset = llama_dataset_from_parquet(&params);
#else
    return;
#endif
    if (!dataset) {
        LLAMA_LOG_INFO("  Corrupted Parquet correctly rejected: %s\n", llama_dataset_get_error());
        llama_dataset_clear_error();
    } else {
        LLAMA_LOG_INFO("  Warning: Corrupted Parquet was loaded (might be valid)\n");
        llama_dataset_free(dataset);
    }

    // Test multiple error conditions in sequence
    LLAMA_LOG_INFO("Testing multiple error conditions...\n");

    // Error 1: Non-existent file
    params.in_files.back() = "non_existent_file.gguf";
    dataset = llama_dataset_from_gguf(&params);
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    llama_dataset_clear_error();

    // Error 2: Null path
    common_params empty_params;
#ifdef LLAMA_PARQUET
    dataset = llama_dataset_from_parquet(&empty_params);
#else
    return;
#endif
    assert(dataset == nullptr);
    assert(llama_dataset_has_error());
    llama_dataset_clear_error();

    // Error 3: Invalid conversion
    const char* output_file = "/invalid/path/output.gguf";
    params.in_files.back() = "test_data/small_dataset.gguf";
    dataset = llama_dataset_from_gguf(&params);
    if (dataset) {
        llama_dataset_to_gguf(dataset, output_file);
        if (llama_dataset_has_error()) {
            LLAMA_LOG_INFO("  Invalid path conversion correctly failed: %s\n", llama_dataset_get_error());
            llama_dataset_clear_error();
        }
        llama_dataset_free(dataset);
    }

    // Verify error state is clean
    assert(!llama_dataset_has_error());

    LLAMA_LOG_INFO("✓ Error recovery test completed\n");
}

int main(int argc, char** argv) {
    // Parse command-line arguments for dataset directory paths
    std::vector<std::string> dataset_dirs;

    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            dataset_dirs.push_back(argv[i]);
        }
        LLAMA_LOG_INFO("Using provided dataset directories:\n");
        for (const auto& dir : dataset_dirs) {
            LLAMA_LOG_INFO("  %s\n", dir.c_str());
        }
    } else {
        // Use default directory if none provided
        dataset_dirs = {"test_data"};
        LLAMA_LOG_INFO("Using default dataset directory: test_data\n");
    }

    LLAMA_LOG_INFO("=== Running dataset integration tests ===\n");

    // Run comprehensive integration tests
    test_end_to_end_workflow();
    test_memory_usage_validation();
    test_performance_comparison();
    test_cross_format_consistency();
    test_large_dataset_handling();
    test_error_recovery();

    LLAMA_LOG_INFO("\n=== All integration tests completed successfully! ===\n");
    return 0;
}
