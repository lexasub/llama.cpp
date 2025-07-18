#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "llama-dataset.h"
#include "streaming-cache.h"
#include "streaming-optimization-manager.h"
#include "streaming-read-ahead.h"

// Helper function to format memory sizes
std::string format_memory_size(size_t bytes);
std::string format_memory_size(size_t bytes) {
    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int suffix_idx = 0;
    double size = bytes;

    while (size >= 1024 && suffix_idx < 4) {
        size /= 1024;
        suffix_idx++;
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << suffixes[suffix_idx] << std::endl;
    return ss.str();
}

// Helper function to measure execution time
template<typename Func>
double measure_time_ms(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Test the new streaming optimization API
void test_streaming_optimization_api();
void test_streaming_optimization_api() {
    LLAMA_LOG_INFO("\n=== Testing Streaming Optimization API ===\n");

    // Load a dataset in streaming mode
    const char* dataset_path = "test_data/small_dataset.gguf";
    common_params params;
    params.in_files.push_back(dataset_path);
    params.dataset_streaming = true;
    struct llama_dataset* dataset = llama_dataset_load_gguf(&params);

    if (!dataset) {
        LLAMA_LOG_ERROR("Failed to load dataset: %s\n", llama_dataset_get_error_message());
        return;
    }

    // Verify streaming is enabled
    bool streaming_enabled = llama_dataset_is_streaming_enabled(dataset);
    LLAMA_LOG_INFO("Streaming enabled: %s\n" , (streaming_enabled ? "yes" : "no"));

    if (!streaming_enabled) {
        LLAMA_LOG_ERROR("Dataset is not in streaming mode\n");
        llama_dataset_free(dataset);
        return;
    }

    // Configure streaming cache size
    bool result = llama_dataset_set_streaming_cache_size(dataset, 32 * 1024 * 1024); // 32MB
    LLAMA_LOG_INFO("Set cache size: %s\n", (result ? "success" : "failed"));

    // Enable read-ahead buffering
    result = llama_dataset_set_streaming_read_ahead(dataset, true, 5);
    LLAMA_LOG_INFO("Enable read-ahead: %s\n", (result ? "success" : "failed"));;

    // Enable adaptive cache sizing
    result = llama_dataset_set_adaptive_cache_sizing(dataset, true);
    LLAMA_LOG_INFO("Enable adaptive cache: %s\n", (result ? "success" : "failed"));;

    // Access some sequences to populate the cache
    uint64_t seq_count = llama_dataset_n_sequences(dataset);
    LLAMA_LOG_INFO("Sequence count: %lu\n", seq_count);

    // Access sequences in order
    for (uint64_t i = 0; i < std::min(seq_count, static_cast<uint64_t>(10)); i++) {
        int32_t seq_len = llama_dataset_sequence_length(dataset, i);
        const int32_t* seq_data = llama_dataset_sequence(dataset, i);

        if (seq_data && seq_len > 0) {
            LLAMA_LOG_INFO("Accessed sequence %lu (length: %d)\n", i, seq_len);
        }
    }

    // Get cache statistics
    double hit_ratio = 0.0;
    size_t memory_usage = 0;
    size_t entry_count = 0;

    result = llama_dataset_get_streaming_stats(dataset, &hit_ratio, &memory_usage, &entry_count);

    if (result) {
        auto memUsage = format_memory_size(memory_usage);
        LLAMA_LOG_INFO("Cache hit ratio: %f\n", hit_ratio);
        LLAMA_LOG_INFO("Memory usage: %s\n", memUsage.c_str());
        LLAMA_LOG_INFO("Cache entries: %lu\n", entry_count);
    } else {
        LLAMA_LOG_ERROR("Failed to get cache statistics: %s\n", llama_dataset_get_error_message());
    }

    // Clean up
    llama_dataset_free(dataset);
}

// Test sequential vs random access performance
void test_access_patterns();
void test_access_patterns() {
    LLAMA_LOG_INFO("\n=== Testing Access Patterns ===\n");

    // Load a dataset in streaming mode
    const char* dataset_path = "test_data/small_dataset.gguf";
    common_params params;
    params.in_files.push_back(dataset_path);
    params.dataset_streaming = true;
    struct llama_dataset* dataset = llama_dataset_load_gguf(&params);

    if (!dataset) {
        LLAMA_LOG_ERROR("Failed to load dataset: %s\n", llama_dataset_get_error_message());
        return;
    }

    uint64_t seq_count = llama_dataset_n_sequences(dataset);
    LLAMA_LOG_INFO("Sequence count: %lu\n", seq_count);

    // Configure streaming optimizations
    llama_dataset_set_streaming_cache_size(dataset, 32 * 1024 * 1024); // 32MB
    llama_dataset_set_streaming_read_ahead(dataset, true, 5);
    llama_dataset_set_adaptive_cache_sizing(dataset, true);

    // Test sequential access
    LLAMA_LOG_INFO("\nTesting sequential access...\n");
    double sequential_time = measure_time_ms([&]() {
        for (uint64_t i = 0; i < std::min(seq_count, static_cast<uint64_t>(20)); i++) {
            llama_dataset_sequence_length(dataset, i);
            llama_dataset_sequence(dataset, i);
        }
    });

    LLAMA_LOG_INFO("Sequential access time: %f ms\n", sequential_time);

    // Get cache statistics after sequential access
    double hit_ratio_seq = 0.0;
    size_t memory_usage_seq = 0;
    size_t entry_count_seq = 0;
    llama_dataset_get_streaming_stats(dataset, &hit_ratio_seq, &memory_usage_seq, &entry_count_seq);
    LLAMA_LOG_INFO("Cache hit ratio (sequential): %f\n", hit_ratio_seq);

    // Reset cache by setting a new size (forces eviction)
    llama_dataset_set_streaming_cache_size(dataset, 32 * 1024 * 1024);

    // Create random access pattern
    std::vector<uint64_t> indices(std::min(seq_count, static_cast<uint64_t>(20)));
    for (uint64_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }

    // Shuffle indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Test random access
    LLAMA_LOG_INFO("\nTesting random access...\n");
    double random_time = measure_time_ms([&]() {
        for (uint64_t i = 0; i < indices.size(); i++) {
            llama_dataset_sequence_length(dataset, indices[i]);
            llama_dataset_sequence(dataset, indices[i]);
        }
    });

    LLAMA_LOG_INFO("Random access time: %f ms\n", random_time);

    // Get cache statistics after random access
    double hit_ratio_rand = 0.0;
    size_t memory_usage_rand = 0;
    size_t entry_count_rand = 0;
    llama_dataset_get_streaming_stats(dataset, &hit_ratio_rand, &memory_usage_rand, &entry_count_rand);
    LLAMA_LOG_INFO("Cache hit ratio (random): %f\n", hit_ratio_rand);

    // Compare performance
    LLAMA_LOG_INFO("\nPerformance comparison:\n");
    LLAMA_LOG_INFO("Sequential vs Random: %f\n x", sequential_time / random_time);
    LLAMA_LOG_INFO("Hit ratio difference: %f\n", hit_ratio_seq - hit_ratio_rand);

    // Clean up
    llama_dataset_free(dataset);
}

// Test memory usage optimization
void test_memory_optimization();
void test_memory_optimization() {
    LLAMA_LOG_INFO("\n=== Testing Memory Usage Optimization ===\n");

    // Load a dataset in streaming mode
    const char* dataset_path = "test_data/small_dataset.gguf";

    // First load without optimization
    LLAMA_LOG_INFO("Loading dataset without optimization...\n");
    common_params params;
    params.in_files.push_back(dataset_path);
    params.dataset_streaming = true;
    struct llama_dataset* dataset1 = llama_dataset_load_gguf(&params);

    if (!dataset1) {
        LLAMA_LOG_ERROR("Failed to load dataset: %s\n", llama_dataset_get_error_message());
        return;
    }

    // Access all sequences to populate cache
    uint64_t seq_count = llama_dataset_n_sequences(dataset1);
    for (uint64_t i = 0; i < seq_count; i++) {
        llama_dataset_sequence_length(dataset1, i);
        llama_dataset_sequence(dataset1, i);
    }

    // Get memory usage without optimization
    double hit_ratio1 = 0.0;
    size_t memory_usage1 = 0;
    size_t entry_count1 = 0;
    llama_dataset_get_streaming_stats(dataset1, &hit_ratio1, &memory_usage1, &entry_count1);

    // Free first dataset
    llama_dataset_free(dataset1);

    // Now load with optimization
    LLAMA_LOG_INFO("Loading dataset with optimization...\n");
    params.in_files.back() = dataset_path;
    params.dataset_streaming = true;
    struct llama_dataset* dataset2 = llama_dataset_load_gguf(&params);

    if (!dataset2) {
        LLAMA_LOG_ERROR("Failed to load dataset: %s\n", llama_dataset_get_error_message());
        return;
    }

    // Enable optimizations
    llama_dataset_set_streaming_cache_size(dataset2, 16 * 1024 * 1024); // Smaller cache
    llama_dataset_set_streaming_read_ahead(dataset2, true, 5);
    llama_dataset_set_adaptive_cache_sizing(dataset2, true);

    // Access all sequences to populate cache
    seq_count = llama_dataset_n_sequences(dataset2);
    for (uint64_t i = 0; i < seq_count; i++) {
        llama_dataset_sequence_length(dataset2, i);
        llama_dataset_sequence(dataset2, i);
    }

    // Get memory usage with optimization
    double hit_ratio2 = 0.0;
    size_t memory_usage2 = 0;
    size_t entry_count2 = 0;
    llama_dataset_get_streaming_stats(dataset2, &hit_ratio2, &memory_usage2, &entry_count2);

    // Compare memory usage
    LLAMA_LOG_INFO("\nMemory usage comparison:\n");
    auto format1 = format_memory_size(memory_usage1);
    auto format2 = format_memory_size(memory_usage2);
    auto format3 = format_memory_size(memory_usage1 - memory_usage2);
    LLAMA_LOG_INFO("Without optimization: %s\n", format1.c_str());
    LLAMA_LOG_INFO("With optimization: %s\n", format2.c_str());
    LLAMA_LOG_INFO("Memory savings: %s (%f %%)\n", format3.c_str(),  100.0 * (memory_usage1 - memory_usage2) / memory_usage1);

    // Clean up
    llama_dataset_free(dataset2);
}

int main() {
    LLAMA_LOG_INFO("Streaming Optimizations Test\n");
    LLAMA_LOG_INFO("==========================\n");

    // Run tests
    test_streaming_optimization_api();
    test_access_patterns();
    test_memory_optimization();

    LLAMA_LOG_INFO("\nAll tests completed!\n");
    return 0;
}
