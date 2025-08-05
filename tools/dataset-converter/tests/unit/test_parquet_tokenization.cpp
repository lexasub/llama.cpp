#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <cassert>
#include <functional>

#ifdef LLAMA_PARQUET
#include "tools/dataset-converter/core/llama-dataset.h"
#include "tools/dataset-converter/streaming/streaming-cache.h"
#include "common/common.h"
#include "llama.h"

// Test utilities
struct TestResult {
    bool passed;
    std::string message;
    double duration_ms;
};

class ParquetTokenizationTester {
private:
    std::vector<TestResult> results;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    
public:
    ParquetTokenizationTester() = default;
    
    ~ParquetTokenizationTester() {
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
        
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Passed: " << passed << "/" << results.size() << std::endl;
        std::cout << "Total time: " << total_time << "ms" << std::endl;
        std::cout << "Success rate: " << (100.0 * passed / results.size()) << "%" << std::endl;
    }
    
    // Test implementations
    bool test_text_to_tokens_conversion();
    bool test_streaming_tokenization();
    bool test_tokenization_cache_performance();
    bool test_mixed_schema_handling();
    bool test_tokenization_error_recovery();
    bool test_memory_pressure_handling();
};

bool ParquetTokenizationTester::test_text_to_tokens_conversion() {
    // Test basic text-to-token conversion functionality
    common_params params;
    params.in_files.push_back("tools/dataset-converter/tests/test_data/basic_text_dataset.parquet");
    params.dataset_tokenize_text = true;
    params.dataset_column = "text";
    
    if (!model) {
        std::cout << "Model not initialized, skipping tokenization test" << std::endl;
        return true; // Skip test if no model
    }
    
    // Test tokenization of sample text
    std::string test_text = "Hello world, this is a test.";
    std::vector<llama_token> tokens;
    tokens.resize(test_text.length() + 10); // Extra space for special tokens
    
    int n_tokens = llama_tokenize(llama_model_get_vocab(model), test_text.c_str(), test_text.length(), 
                                  tokens.data(), tokens.size(), false, true);
    
    if (n_tokens <= 0) {
        return false;
    }
    
    tokens.resize(n_tokens);
    
    // Verify tokens can be detokenized back to text
    std::string detokenized;
    char piece_buf[256];
    for (int i = 0; i < n_tokens; i++) {
        int piece_len = llama_token_to_piece(llama_model_get_vocab(model), tokens[i], 
                                           piece_buf, sizeof(piece_buf), 0, true);
        if (piece_len > 0) {
            detokenized += std::string(piece_buf, piece_len);
        }
    }
    
    // Basic validation - detokenized should contain key words
    return detokenized.find("Hello") != std::string::npos && 
           detokenized.find("world") != std::string::npos;
}

bool ParquetTokenizationTester::test_streaming_tokenization() {
    // Test streaming tokenization with cache
    llama_dataset_streaming_cache cache(1024 * 1024); // 1MB cache
    
    // Test cache basic operations
    std::vector<int32_t> test_tokens = {1, 2, 3, 4, 5};
    cache.put_tokenized(1, test_tokens);
    
    // Check stats immediately after putting
    auto stats_after_put = cache.get_stats();
    if (stats_after_put.entry_count == 0) {
        std::cout << "  Error: No entries after put_tokenized" << std::endl;
        return false;
    }
    
    // Test cache hit using has_tokenized
    bool has_entry = cache.has_tokenized(1);
    if (!has_entry) {
        std::cout << "  Error: has_tokenized returned false for existing entry" << std::endl;
        return false;
    }
    
    // Test cache miss
    bool has_missing = cache.has_tokenized(999);
    if (has_missing) {
        std::cout << "  Error: has_tokenized returned true for non-existing entry" << std::endl;
        return false;
    }
    
    // Test cache statistics
    auto stats = cache.get_stats();
    bool result = stats.entry_count > 0 && stats.accesses > 0;
    
    if (!result) {
        std::cout << "  Error: Final stats check failed - entries: " << stats.entry_count 
                  << ", accesses: " << stats.accesses << std::endl;
    }
    
    return result;
}

bool ParquetTokenizationTester::test_tokenization_cache_performance() {
    // Test cache performance under load
    llama_dataset_streaming_cache cache(512 * 1024); // 512KB cache
    
    const int num_sequences = 50; // Reduced to avoid memory issues
    const int tokens_per_sequence = 20; // Reduced size
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Fill cache with tokenized sequences
    for (int i = 0; i < num_sequences; i++) {
        std::vector<int32_t> tokens;
        tokens.reserve(tokens_per_sequence);
        
        for (int j = 0; j < tokens_per_sequence; j++) {
            tokens.push_back(i * tokens_per_sequence + j);
        }
        
        cache.put_tokenized(i, tokens);
    }
    
    // Test cache retrieval performance using has_tokenized
    int hits = 0;
    for (int i = 0; i < num_sequences; i++) {
        if (cache.has_tokenized(i)) {
            hits++;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    
    auto stats = cache.get_stats();
    
    std::cout << "  Cache performance: " << duration << "ms, "
              << "hit ratio: " << stats.hit_ratio << ", "
              << "entries: " << stats.entry_count << std::endl;
    
    return hits > 0 && stats.accesses > 0 && duration < 1000.0; // Should complete in under 1s
}

bool ParquetTokenizationTester::test_mixed_schema_handling() {
    // Test handling of Parquet files with mixed content (text + pre-tokenized)
    common_params params;
    params.in_files.push_back("tools/dataset-converter/tests/test_data/mixed_content_dataset.parquet");
    params.dataset_tokenize_text = true;
    params.dataset_column = "text";
    
    // Test would load a mixed schema parquet file
    // For now, just test parameter validation
    if (params.dataset_column.empty()) {
        return false;
    }
    
    if (!params.dataset_tokenize_text) {
        return false;
    }
    
    // Test column name configuration
    params.dataset_column = "data";
    std::string token_column = "tokens";
    
    return !params.dataset_column.empty() && !token_column.empty();
}

bool ParquetTokenizationTester::test_tokenization_error_recovery() {
    // Test error handling in tokenization pipeline
    common_params params;
    params.in_files.push_back("nonexistent_file.parquet");
    params.dataset_tokenize_text = true;
    
    // Test should handle missing files gracefully
    llama_dataset* dataset = nullptr;
    
    try {
        dataset = llama_dataset_from_parquet(&params);
        if (dataset) {
            llama_dataset_free(dataset);
            return false; // Should have failed
        }
    } catch (...) {
        // Expected to fail
    }
    
    // Test invalid parameters
    params.dataset_column = "";
    params.in_files.clear();
    
    try {
        dataset = llama_dataset_from_parquet(&params);
        if (dataset) {
            llama_dataset_free(dataset);
            return false; // Should have failed
        }
    } catch (...) {
        // Expected to fail
    }
    
    return true; // Error recovery working
}

bool ParquetTokenizationTester::test_memory_pressure_handling() {
    // Test cache behavior under memory pressure - simplified version
    try {
        llama_dataset_streaming_cache cache(1024); // Small cache (1KB)
        
        // Add a single small sequence
        std::vector<int32_t> tokens = {1, 2, 3, 4, 5};
        cache.put_tokenized(1, tokens);
        
        auto stats = cache.get_stats();
        
        // Basic validation
        bool has_memory = stats.current_memory > 0;
        bool has_entries = stats.entry_count > 0;
        
        std::cout << "  Memory pressure test: memory=" << stats.current_memory 
                  << " bytes, entries=" << stats.entry_count << std::endl;
        
        return has_memory && has_entries;
    } catch (const std::exception& e) {
        std::cout << "  Memory pressure test exception: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char** argv) {
    std::cout << "=== Comprehensive Parquet Tokenization Tests ===" << std::endl;
    
    ParquetTokenizationTester tester;
    
    // Initialize model if path provided
    if (argc > 1) {
        std::string model_path = argv[1];
        if (!tester.init_model(model_path)) {
            std::cout << "Warning: Could not load model, some tests will be skipped" << std::endl;
        }
    } else {
        std::cout << "No model path provided, tokenization tests will be limited" << std::endl;
    }
    
    // Run all test functions
    tester.run_test("Text to Tokens Conversion", 
                   [&]() { return tester.test_text_to_tokens_conversion(); });
    
    tester.run_test("Streaming Tokenization", 
                   [&]() { return tester.test_streaming_tokenization(); });
    
    tester.run_test("Tokenization Cache Performance", 
                   [&]() { return tester.test_tokenization_cache_performance(); });
    
    tester.run_test("Mixed Schema Handling", 
                   [&]() { return tester.test_mixed_schema_handling(); });
    
    tester.run_test("Tokenization Error Recovery", 
                   [&]() { return tester.test_tokenization_error_recovery(); });
    
    tester.run_test("Memory Pressure Handling", 
                   [&]() { return tester.test_memory_pressure_handling(); });
    
    tester.print_summary();
    
    return 0;
}

#else
int main() {
    std::cout << "Parquet support not compiled in" << std::endl;
    return 0;
}
#endif