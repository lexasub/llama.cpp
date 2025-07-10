#include "llama-dataset.h"
#include "streaming-optimization-manager.h"
#include "streaming-cache.h"
#include "streaming-read-ahead.h"
#include "streaming-memory-monitor.h"

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <random>
#include <thread>

// Helper function to format memory sizes
std::string format_memory_size(size_t bytes) {
    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int suffix_idx = 0;
    double size = bytes;

    while (size >= 1024 && suffix_idx < 4) {
        size /= 1024;
        suffix_idx++;
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << suffixes[suffix_idx];
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
void test_streaming_optimization_api() {
    std::cout << "\n=== Testing Streaming Optimization API ===" << std::endl;

    // Load a dataset in streaming mode
    const char* dataset_path = "test_data/small_dataset.gguf";
    struct llama_dataset* dataset = llama_dataset_load_gguf(dataset_path, true);

    if (!dataset) {
        std::cerr << "Failed to load dataset: " << llama_dataset_get_error_message() << std::endl;
        return;
    }

    // Verify streaming is enabled
    bool streaming_enabled = llama_dataset_is_streaming_enabled(dataset);
    std::cout << "Streaming enabled: " << (streaming_enabled ? "yes" : "no") << std::endl;

    if (!streaming_enabled) {
        std::cerr << "Dataset is not in streaming mode" << std::endl;
        llama_dataset_free(dataset);
        return;
    }

    // Configure streaming cache size
    bool result = llama_dataset_set_streaming_cache_size(dataset, 32 * 1024 * 1024); // 32MB
    std::cout << "Set cache size: " << (result ? "success" : "failed") << std::endl;

    // Enable read-ahead buffering
    result = llama_dataset_set_streaming_read_ahead(dataset, true, 5);
    std::cout << "Enable read-ahead: " << (result ? "success" : "failed") << std::endl;

    // Enable adaptive cache sizing
    result = llama_dataset_set_adaptive_cache_sizing(dataset, true);
    std::cout << "Enable adaptive cache: " << (result ? "success" : "failed") << std::endl;

    // Access some sequences to populate the cache
    uint64_t seq_count = n_sequences(dataset);
    std::cout << "Sequence count: " << seq_count << std::endl;

    // Access sequences in order
    for (uint64_t i = 0; i < std::min(seq_count, (uint64_t)10); i++) {
        int32_t seq_len = sequence_length(dataset, i);
        const int32_t* seq_data = sequence(dataset, i);

        if (seq_data && seq_len > 0) {
            std::cout << "Accessed sequence " << i << " (length: " << seq_len << ")" << std::endl;
        }
    }

    // Get cache statistics
    double hit_ratio = 0.0;
    size_t memory_usage = 0;
    size_t entry_count = 0;

    result = llama_dataset_get_streaming_stats(dataset, &hit_ratio, &memory_usage, &entry_count);

    if (result) {
        std::cout << "Cache hit ratio: " << hit_ratio << std::endl;
        std::cout << "Memory usage: " << format_memory_size(memory_usage) << std::endl;
        std::cout << "Cache entries: " << entry_count << std::endl;
    } else {
        std::cerr << "Failed to get cache statistics: " << llama_dataset_get_error_message() << std::endl;
    }

    // Clean up
    llama_dataset_free(dataset);
}

// Test sequential vs random access performance
void test_access_patterns() {
    std::cout << "\n=== Testing Access Patterns ===" << std::endl;

    // Load a dataset in streaming mode
    const char* dataset_path = "test_data/small_dataset.gguf";
    struct llama_dataset* dataset = llama_dataset_load_gguf(dataset_path, true);

    if (!dataset) {
        std::cerr << "Failed to load dataset: " << llama_dataset_get_error_message() << std::endl;
        return;
    }

    uint64_t seq_count = n_sequences(dataset);
    std::cout << "Sequence count: " << seq_count << std::endl;

    // Configure streaming optimizations
    llama_dataset_set_streaming_cache_size(dataset, 32 * 1024 * 1024); // 32MB
    llama_dataset_set_streaming_read_ahead(dataset, true, 5);
    llama_dataset_set_adaptive_cache_sizing(dataset, true);

    // Test sequential access
    std::cout << "\nTesting sequential access..." << std::endl;
    double sequential_time = measure_time_ms([&]() {
        for (uint64_t i = 0; i < std::min(seq_count, (uint64_t)20); i++) {
            sequence_length(dataset, i);
            sequence(dataset, i);
        }
    });

    std::cout << "Sequential access time: " << sequential_time << " ms" << std::endl;

    // Get cache statistics after sequential access
    double hit_ratio_seq = 0.0;
    size_t memory_usage_seq = 0;
    size_t entry_count_seq = 0;
    llama_dataset_get_streaming_stats(dataset, &hit_ratio_seq, &memory_usage_seq, &entry_count_seq);
    std::cout << "Cache hit ratio (sequential): " << hit_ratio_seq << std::endl;

    // Reset cache by setting a new size (forces eviction)
    llama_dataset_set_streaming_cache_size(dataset, 32 * 1024 * 1024);

    // Create random access pattern
    std::vector<uint64_t> indices(std::min(seq_count, (uint64_t)20));
    for (uint64_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }

    // Shuffle indices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Test random access
    std::cout << "\nTesting random access..." << std::endl;
    double random_time = measure_time_ms([&]() {
        for (uint64_t i = 0; i < indices.size(); i++) {
            sequence_length(dataset, indices[i]);
            sequence(dataset, indices[i]);
        }
    });

    std::cout << "Random access time: " << random_time << " ms" << std::endl;

    // Get cache statistics after random access
    double hit_ratio_rand = 0.0;
    size_t memory_usage_rand = 0;
    size_t entry_count_rand = 0;
    llama_dataset_get_streaming_stats(dataset, &hit_ratio_rand, &memory_usage_rand, &entry_count_rand);
    std::cout << "Cache hit ratio (random): " << hit_ratio_rand << std::endl;

    // Compare performance
    std::cout << "\nPerformance comparison:" << std::endl;
    std::cout << "Sequential vs Random: " << (sequential_time / random_time) << "x" << std::endl;
    std::cout << "Hit ratio difference: " << (hit_ratio_seq - hit_ratio_rand) << std::endl;

    // Clean up
    llama_dataset_free(dataset);
}

// Test memory usage optimization
void test_memory_optimization() {
    std::cout << "\n=== Testing Memory Usage Optimization ===" << std::endl;

    // Load a dataset in streaming mode
    const char* dataset_path = "test_data/small_dataset.gguf";

    // First load without optimization
    std::cout << "Loading dataset without optimization..." << std::endl;
    struct llama_dataset* dataset1 = llama_dataset_load_gguf(dataset_path, true);

    if (!dataset1) {
        std::cerr << "Failed to load dataset: " << llama_dataset_get_error_message() << std::endl;
        return;
    }

    // Access all sequences to populate cache
    uint64_t seq_count = n_sequences(dataset1);
    for (uint64_t i = 0; i < seq_count; i++) {
        sequence_length(dataset1, i);
        sequence(dataset1, i);
    }

    // Get memory usage without optimization
    double hit_ratio1 = 0.0;
    size_t memory_usage1 = 0;
    size_t entry_count1 = 0;
    llama_dataset_get_streaming_stats(dataset1, &hit_ratio1, &memory_usage1, &entry_count1);

    // Free first dataset
    llama_dataset_free(dataset1);

    // Now load with optimization
    std::cout << "Loading dataset with optimization..." << std::endl;
    struct llama_dataset* dataset2 = llama_dataset_load_gguf(dataset_path, true);

    if (!dataset2) {
        std::cerr << "Failed to load dataset: " << llama_dataset_get_error_message() << std::endl;
        return;
    }

    // Enable optimizations
    llama_dataset_set_streaming_cache_size(dataset2, 16 * 1024 * 1024); // Smaller cache
    llama_dataset_set_streaming_read_ahead(dataset2, true, 5);
    llama_dataset_set_adaptive_cache_sizing(dataset2, true);

    // Access all sequences to populate cache
    seq_count = n_sequences(dataset2);
    for (uint64_t i = 0; i < seq_count; i++) {
        sequence_length(dataset2, i);
        sequence(dataset2, i);
    }

    // Get memory usage with optimization
    double hit_ratio2 = 0.0;
    size_t memory_usage2 = 0;
    size_t entry_count2 = 0;
    llama_dataset_get_streaming_stats(dataset2, &hit_ratio2, &memory_usage2, &entry_count2);

    // Compare memory usage
    std::cout << "\nMemory usage comparison:" << std::endl;
    std::cout << "Without optimization: " << format_memory_size(memory_usage1) << std::endl;
    std::cout << "With optimization: " << format_memory_size(memory_usage2) << std::endl;
    std::cout << "Memory savings: " << format_memory_size(memory_usage1 - memory_usage2)
              << " (" << (100.0 * (memory_usage1 - memory_usage2) / memory_usage1) << "%)" << std::endl;

    // Clean up
    llama_dataset_free(dataset2);
}

int main() {
    std::cout << "Streaming Optimizations Test" << std::endl;
    std::cout << "==========================" << std::endl;

    // Run tests
    test_streaming_optimization_api();
    test_access_patterns();
    test_memory_optimization();

    std::cout << "\nAll tests completed!" << std::endl;
    return 0;
}
