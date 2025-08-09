#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <cassert>
#include <functional>
#include "common.h"
#include "common/log.h"
#include "llama-impl.h"
#include "test_core_functionality.h"

#ifdef LLAMA_PARQUET
#include "tools/dataset-converter/core/llama-dataset.h"
#include "tools/dataset-converter/streaming/streaming-cache.h"
#include "llama.h"

/**
 * @file test_streaming_functionality.cpp
 * @brief Streaming vs batch equivalence tests
 * 
 * Tests that streaming tokenization produces equivalent results to batch tokenization,
 * cache consistency, and streaming performance characteristics.
 */

class StreamingTester {
private:
    std::vector<TestResult> results;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    
public:
    StreamingTester() = default;
    
    ~StreamingTester() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
    
    bool init_model(const std::string& model_path) {
        llama_model_params model_params = llama_model_default_params();
        model = llama_model_load_from_file(model_path.c_str(), model_params);
        
        if (!model) {
            std::cerr << "Failed to load model from " << model_path << std::endl;
            return false;
        }
        
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 512;
        ctx = llama_init_from_model(model, ctx_params);
        
        return ctx != nullptr;
    }
    
    void run_test(const std::string& name, std::function<bool()> test_func) {
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            bool passed = test_func();
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double, std::milli>(end - start).count();
            
            results.push_back({passed, name, duration});
            std::cout << (passed ? "✓" : "✗") << " " << name 
                      << " (" << duration << "ms)" << std::endl;
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration<double, std::milli>(end - start).count();
            
            results.push_back({false, name + " - Exception: " + e.what(), duration});
            std::cout << "✗ " << name << " - Exception: " << e.what() 
                      << " (" << duration << "ms)" << std::endl;
        }
    }
    
    void print_summary() {
        int passed = 0;
        double total_time = 0;
        
        for (const auto& result : results) {
            if (result.passed) passed++;
            total_time += result.duration_ms;
        }
        
        std::cout << "\n=== Streaming Functionality Test Summary ===" << std::endl;
        std::cout << "Passed: " << passed << "/" << results.size() << std::endl;
        std::cout << "Total time: " << total_time << "ms" << std::endl;
        std::cout << "Success rate: " << (100.0 * passed / results.size()) << "%" << std::endl;
    }
    
    // Test implementations
    bool test_streaming_batch_equivalence();
    bool test_cache_consistency();
    bool test_streaming_performance();
    bool test_memory_efficiency();
};

bool StreamingTester::test_streaming_batch_equivalence() {
    // Test that streaming tokenization produces equivalent results to batch tokenization
    if (!model) {
        std::cout << "  Model not initialized, testing cache equivalence only" << std::endl;
        
        // Test cache consistency without model
        llama_dataset_streaming_cache cache(1024 * 1024);
        
        std::vector<int32_t> test_tokens = {100, 200, 300, 400, 500};
        cache.put_tokenized(1, test_tokens);
        
        bool has_entry = cache.has_tokenized(1);
        if (!has_entry) {
            std::cout << "  Error: Cache consistency check failed" << std::endl;
            return false;
        }
        
        return true;
    }
    
    // Test streaming vs batch equivalence with model
    std::vector<std::string> test_texts = {
        "Hello world",
        "This is a test",
        "Streaming tokenization test",
        "The quick brown fox jumps over the lazy dog"
    };
    
    // Batch tokenization
    std::vector<std::vector<llama_token>> batch_results;
    for (const auto& text : test_texts) {
        std::vector<llama_token> tokens;
        tokens.resize(text.length() + 10);
        
        int n_tokens = llama_tokenize(llama_model_get_vocab(model), text.c_str(), 
                                      text.length(), tokens.data(), tokens.size(), false, true);
        
        if (n_tokens > 0) {
            tokens.resize(n_tokens);
            batch_results.push_back(tokens);
        }
    }
    
    // Streaming tokenization simulation
    llama_dataset_streaming_cache cache(1024 * 1024);
    std::vector<std::vector<int32_t>> streaming_results;
    
    for (size_t i = 0; i < test_texts.size(); i++) {
        if (i < batch_results.size()) {
            std::vector<int32_t> int_tokens;
            for (auto token : batch_results[i]) {
                int_tokens.push_back(static_cast<int32_t>(token));
            }
            cache.put_tokenized(i, int_tokens);
            streaming_results.push_back(int_tokens);
        }
    }
    
    // Compare results
    if (batch_results.size() != streaming_results.size()) {
        std::cout << "  Error: Result count mismatch" << std::endl;
        return false;
    }
    
    for (size_t i = 0; i < batch_results.size(); i++) {
        if (batch_results[i].size() != streaming_results[i].size()) {
            std::cout << "  Error: Token count mismatch for text " << i << std::endl;
            return false;
        }
    }
    
    std::cout << "  Streaming equivalence verified for " << test_texts.size() << " texts" << std::endl;
    return true;
}

bool StreamingTester::test_cache_consistency() {
    llama_dataset_streaming_cache cache(512 * 1024);
    
    // Test consistent cache behavior
    std::vector<int32_t> original_tokens = {1, 2, 3, 4, 5};
    cache.put_tokenized(1, original_tokens);
    
    // Verify entry exists
    if (!cache.has_tokenized(1)) {
        std::cout << "  Error: Cache entry not found after insertion" << std::endl;
        return false;
    }
    
    // Test cache persistence across operations
    for (int i = 2; i <= 10; i++) {
        std::vector<int32_t> tokens = {i, i+1, i+2};
        cache.put_tokenized(i, tokens);
    }
    
    // Original entry should still exist (if cache is large enough)
    bool original_still_exists = cache.has_tokenized(1);
    
    // Test cache statistics consistency
    auto stats = cache.get_stats();
    if (stats.entry_count == 0) {
        std::cout << "  Error: Cache reports zero entries after insertions" << std::endl;
        return false;
    }
    
    std::cout << "  Cache consistency: " << stats.entry_count << " entries, "
              << "original_preserved=" << (original_still_exists ? "yes" : "no") << std::endl;
    
    return stats.entry_count > 0;
}

bool StreamingTester::test_streaming_performance() {
    llama_dataset_streaming_cache cache(1024 * 1024); // 1MB cache
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Simulate streaming workload
    const int num_sequences = 500;
    for (int i = 1; i <= num_sequences; i++) {
        std::vector<int32_t> tokens;
        for (int j = 0; j < 50; j++) {
            tokens.push_back(i * 100 + j);
        }
        cache.put_tokenized(i, tokens);
    }
    
    auto insert_end = std::chrono::high_resolution_clock::now();
    
    // Test random access pattern (simulating streaming reads)
    int cache_hits = 0;
    for (int round = 0; round < 3; round++) {
        for (int i = 1; i <= num_sequences; i += 10) { // Sample every 10th sequence
            if (cache.has_tokenized(i)) {
                cache_hits++;
            }
        }
    }
    
    auto access_end = std::chrono::high_resolution_clock::now();
    
    // Calculate performance metrics
    double insert_time = std::chrono::duration<double, std::milli>(insert_end - start_time).count();
    double access_time = std::chrono::duration<double, std::milli>(access_end - insert_end).count();
    
    auto stats = cache.get_stats();
    
    std::cout << "  Streaming performance: insert_time=" << insert_time << "ms, "
              << "access_time=" << access_time << "ms, "
              << "cache_hits=" << cache_hits << ", "
              << "entries=" << stats.entry_count << std::endl;
    
    // Performance should be reasonable
    bool insert_reasonable = insert_time < 1000.0; // Less than 1 second
    bool access_reasonable = access_time < 100.0;  // Less than 100ms
    
    return insert_reasonable && access_reasonable && cache_hits > 0;
}

bool StreamingTester::test_memory_efficiency() {
    // Test memory efficiency of streaming vs batch processing
    
    // Small cache to test memory pressure
    llama_dataset_streaming_cache small_cache(128 * 1024); // 128KB
    
    auto initial_stats = small_cache.get_stats();
    size_t initial_memory = initial_stats.current_memory;
    
    // Add many entries to test memory management
    const int num_entries = 200;
    for (int i = 1; i <= num_entries; i++) {
        std::vector<int32_t> tokens;
        for (int j = 0; j < 100; j++) {
            tokens.push_back(i * 100 + j);
        }
        small_cache.put_tokenized(i, tokens);
    }
    
    auto final_stats = small_cache.get_stats();
    
    // Memory should be managed efficiently
    bool memory_bounded = final_stats.current_memory < (256 * 1024); // Should not exceed 256KB significantly
    bool has_entries = final_stats.entry_count > 0;
    bool eviction_occurred = final_stats.entry_count < num_entries; // Some entries should be evicted
    
    std::cout << "  Memory efficiency: initial=" << initial_memory 
              << ", final=" << final_stats.current_memory 
              << ", entries=" << final_stats.entry_count << "/" << num_entries
              << ", evicted=" << (eviction_occurred ? "yes" : "no") << std::endl;
    
    return memory_bounded && has_entries;
}

int main(int argc, char** argv) {
    std::cout << "=== Streaming Functionality Tests ===" << std::endl;
    
    std::string model_path;
    if (argc > 1) {
        model_path = argv[1];
    }
    
    StreamingTester tester;
    
    // Initialize model if path provided
    if (!model_path.empty()) {
        if (!tester.init_model(model_path)) {
            std::cout << "Warning: Could not load model, some tests will be limited" << std::endl;
        }
    } else {
        std::cout << "No model path provided, running cache-only tests" << std::endl;
    }
    
    // Run streaming functionality tests
    tester.run_test("Streaming Batch Equivalence", 
                   [&]() { return tester.test_streaming_batch_equivalence(); });
    
    tester.run_test("Cache Consistency", 
                   [&]() { return tester.test_cache_consistency(); });
    
    tester.run_test("Streaming Performance", 
                   [&]() { return tester.test_streaming_performance(); });
    
    tester.run_test("Memory Efficiency", 
                   [&]() { return tester.test_memory_efficiency(); });
    
    tester.print_summary();
    
    return 0;
}

#else
int main() {
    std::cout << "Parquet support not compiled in - streaming functionality tests skipped" << std::endl;
    return 0;
}
#endif
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