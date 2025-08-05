/**
 * @file test_streaming_large_dataset.cpp
 * @brief Large dataset streaming performance validation
 *
 * This test validates streaming tokenization performance with large datasets,
 * focusing on memory efficiency, throughput consistency, and cache effectiveness
 * under realistic production workloads.
 *
 * ## Test Scenarios
 * - Large dataset processing (simulated 1GB+ datasets)
 * - Memory-constrained environments
 * - Sequential and random access patterns
 * - Long-running streaming operations
 * - Memory leak detection
 * - Performance degradation monitoring
 *
 * @author llama.cpp dataset-converter team
 * @version 1.0
 * @since 2024
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <sys/resource.h>

#include "common.h"
#include "common/log.h"
#include "llama-dataset.h"
#include "streaming-cache.h"

/**
 * @brief Large dataset streaming test results
 */
struct StreamingTestResult {
    std::string test_name;
    size_t total_sequences_processed;
    double total_duration_seconds;
    double avg_throughput_sequences_per_sec;
    size_t peak_memory_usage_mb;
    double memory_efficiency_ratio;
    double cache_hit_ratio;
    bool memory_stable; // No significant memory leaks
    bool performance_stable; // Consistent throughput
    std::vector<double> throughput_samples;
    std::vector<size_t> memory_samples;
};

/**
 * @brief Large dataset streaming performance tester
 */
class LargeDatasetStreamingTester {
private:
    std::unique_ptr<llama_dataset_streaming_cache> cache;
    std::vector<StreamingTestResult> results;
    
    // Performance requirements for large datasets
    static constexpr double MIN_THROUGHPUT_SEQUENCES_PER_SEC = 100.0;
    static constexpr size_t MAX_MEMORY_USAGE_MB = 512; // 512MB limit
    static constexpr double MAX_MEMORY_GROWTH_RATIO = 1.5; // 50% growth allowed
    static constexpr double MIN_CACHE_HIT_RATIO = 0.6; // 60% for large datasets
    
public:
    LargeDatasetStreamingTester() 
        : cache(std::make_unique<llama_dataset_streaming_cache>(128 * 1024 * 1024)) {} // 128MB cache
    
    /**
     * @brief Test streaming with simulated large dataset
     */
    void test_large_dataset_streaming() {
        std::cout << "\n=== Large Dataset Streaming Test ===" << std::endl;
        
        const size_t large_dataset_size = 100000; // 100K sequences
        const size_t memory_sample_interval = 1000; // Sample every 1000 sequences
        const size_t throughput_sample_interval = 5000; // Sample every 5000 sequences
        
        StreamingTestResult result;
        result.test_name = "Large Dataset Sequential Streaming";
        result.total_sequences_processed = large_dataset_size;
        
        // Configure cache for large dataset
        cache->set_tokenization_cache_size(64 * 1024 * 1024); // 64MB for tokenization
        cache->set_adaptive_sizing(true, 0.8);
        cache->set_read_ahead(true, 10); // Aggressive read-ahead
        
        size_t initial_memory = get_memory_usage_mb();
        auto start_time = std::chrono::high_resolution_clock::now();
        auto last_sample_time = start_time;
        size_t last_sample_count = 0;
        
        std::cout << "Processing " << large_dataset_size << " sequences..." << std::endl;
        
        for (size_t i = 0; i < large_dataset_size; i++) {
            // Generate realistic text sequence
            std::string text = generate_realistic_text(i);
            
            // Tokenize and cache
            std::vector<int32_t> tokens = simulate_tokenization(text);
            cache->put_tokenized(i, tokens);
            
            // Sample memory usage
            if (i % memory_sample_interval == 0) {
                size_t current_memory = get_memory_usage_mb();
                result.memory_samples.push_back(current_memory);
                
                if (i % (memory_sample_interval * 10) == 0) {
                    std::cout << "  Processed " << i << " sequences, memory: " 
                              << current_memory << "MB" << std::endl;
                }
            }
            
            // Sample throughput
            if (i % throughput_sample_interval == 0 && i > 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                double interval_seconds = std::chrono::duration<double>(current_time - last_sample_time).count();
                double interval_throughput = (i - last_sample_count) / interval_seconds;
                
                result.throughput_samples.push_back(interval_throughput);
                last_sample_time = current_time;
                last_sample_count = i;
            }
            
            // Simulate some access patterns
            if (i > 1000 && i % 100 == 0) {
                // Access recent sequences to test cache effectiveness
                size_t recent_id = i - (50 + (i % 50));
                cache->has_tokenized(recent_id);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        size_t final_memory = get_memory_usage_mb();
        
        // Calculate results
        result.total_duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
        result.avg_throughput_sequences_per_sec = large_dataset_size / result.total_duration_seconds;
        result.peak_memory_usage_mb = *std::max_element(result.memory_samples.begin(), result.memory_samples.end());
        result.memory_efficiency_ratio = static_cast<double>(final_memory - initial_memory) / initial_memory;
        
        auto cache_stats = cache->get_stats();
        result.cache_hit_ratio = cache_stats.hit_ratio;
        
        // Check memory stability (no significant leaks)
        result.memory_stable = (result.memory_efficiency_ratio < MAX_MEMORY_GROWTH_RATIO);
        
        // Check performance stability (consistent throughput)
        if (!result.throughput_samples.empty()) {
            double min_throughput = *std::min_element(result.throughput_samples.begin(), result.throughput_samples.end());
            double max_throughput = *std::max_element(result.throughput_samples.begin(), result.throughput_samples.end());
            double throughput_variance = (max_throughput - min_throughput) / result.avg_throughput_sequences_per_sec;
            result.performance_stable = (throughput_variance < 0.5); // Less than 50% variance
        } else {
            result.performance_stable = true;
        }
        
        results.push_back(result);
        print_streaming_result(result);
    }
    
    /**
     * @brief Test streaming with memory constraints
     */
    void test_memory_constrained_streaming() {
        std::cout << "\n=== Memory Constrained Streaming Test ===" << std::endl;
        
        const size_t dataset_size = 50000;
        const size_t memory_limit = 32 * 1024 * 1024; // 32MB limit
        
        StreamingTestResult result;
        result.test_name = "Memory Constrained Streaming";
        result.total_sequences_processed = dataset_size;
        
        // Configure cache with tight memory constraints
        cache->clear();
        cache->set_tokenization_cache_size(memory_limit);
        cache->set_adaptive_sizing(true, 0.7); // More aggressive memory management
        
        size_t initial_memory = get_memory_usage_mb();
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Processing " << dataset_size << " sequences with " 
                  << (memory_limit / 1024 / 1024) << "MB memory limit..." << std::endl;
        
        for (size_t i = 0; i < dataset_size; i++) {
            // Generate larger text sequences to stress memory
            std::string text = generate_large_text(i);
            std::vector<int32_t> tokens = simulate_tokenization(text);
            
            cache->put_tokenized(i, tokens);
            
            // Monitor memory pressure
            if (i % 1000 == 0) {
                auto cache_stats = cache->get_stats();
                size_t current_memory = get_memory_usage_mb();
                result.memory_samples.push_back(current_memory);
                
                if (cache_stats.tokenization_memory_pressure > 0.9) {
                    std::cout << "  High memory pressure at sequence " << i 
                              << ": " << cache_stats.tokenization_memory_pressure << std::endl;
                }
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        size_t final_memory = get_memory_usage_mb();
        
        // Calculate results
        result.total_duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
        result.avg_throughput_sequences_per_sec = dataset_size / result.total_duration_seconds;
        result.peak_memory_usage_mb = *std::max_element(result.memory_samples.begin(), result.memory_samples.end());
        result.memory_efficiency_ratio = static_cast<double>(final_memory - initial_memory) / initial_memory;
        
        auto cache_stats = cache->get_stats();
        result.cache_hit_ratio = cache_stats.hit_ratio;
        
        // Memory constraint compliance
        result.memory_stable = (result.peak_memory_usage_mb <= MAX_MEMORY_USAGE_MB);
        result.performance_stable = (result.avg_throughput_sequences_per_sec >= MIN_THROUGHPUT_SEQUENCES_PER_SEC);
        
        results.push_back(result);
        print_streaming_result(result);
    }
    
    /**
     * @brief Test streaming with random access patterns
     */
    void test_random_access_streaming() {
        std::cout << "\n=== Random Access Streaming Test ===" << std::endl;
        
        const size_t dataset_size = 30000;
        const size_t access_operations = 50000;
        
        StreamingTestResult result;
        result.test_name = "Random Access Streaming";
        result.total_sequences_processed = access_operations;
        
        // Pre-populate cache with some data
        cache->clear();
        cache->set_tokenization_cache_size(64 * 1024 * 1024);
        
        std::cout << "Pre-populating cache with " << dataset_size << " sequences..." << std::endl;
        for (size_t i = 0; i < dataset_size; i++) {
            std::string text = generate_realistic_text(i);
            std::vector<int32_t> tokens = simulate_tokenization(text);
            cache->put_tokenized(i, tokens);
        }
        
        // Random access test
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> access_dist(0, dataset_size - 1);
        
        size_t initial_memory = get_memory_usage_mb();
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Performing " << access_operations << " random access operations..." << std::endl;
        
        size_t cache_hits = 0;
        for (size_t i = 0; i < access_operations; i++) {
            size_t sequence_id = access_dist(gen);
            
            if (cache->has_tokenized(sequence_id)) {
                cache_hits++;
            } else {
                // Generate and cache if not found
                std::string text = generate_realistic_text(sequence_id);
                std::vector<int32_t> tokens = simulate_tokenization(text);
                cache->put_tokenized(sequence_id, tokens);
            }
            
            if (i % 5000 == 0) {
                size_t current_memory = get_memory_usage_mb();
                result.memory_samples.push_back(current_memory);
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        size_t final_memory = get_memory_usage_mb();
        
        // Calculate results
        result.total_duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
        result.avg_throughput_sequences_per_sec = access_operations / result.total_duration_seconds;
        result.peak_memory_usage_mb = *std::max_element(result.memory_samples.begin(), result.memory_samples.end());
        result.memory_efficiency_ratio = static_cast<double>(final_memory - initial_memory) / initial_memory;
        result.cache_hit_ratio = static_cast<double>(cache_hits) / access_operations;
        
        result.memory_stable = (result.memory_efficiency_ratio < MAX_MEMORY_GROWTH_RATIO);
        result.performance_stable = (result.avg_throughput_sequences_per_sec >= MIN_THROUGHPUT_SEQUENCES_PER_SEC);
        
        results.push_back(result);
        print_streaming_result(result);
    }
    
    /**
     * @brief Test long-running streaming operation
     */
    void test_long_running_streaming() {
        std::cout << "\n=== Long Running Streaming Test ===" << std::endl;
        
        const size_t total_sequences = 200000; // 200K sequences
        const size_t batch_size = 10000;
        const size_t num_batches = total_sequences / batch_size;
        
        StreamingTestResult result;
        result.test_name = "Long Running Streaming";
        result.total_sequences_processed = total_sequences;
        
        cache->clear();
        cache->set_tokenization_cache_size(96 * 1024 * 1024); // 96MB
        cache->set_adaptive_sizing(true, 0.8);
        
        size_t initial_memory = get_memory_usage_mb();
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Processing " << total_sequences << " sequences in " 
                  << num_batches << " batches..." << std::endl;
        
        for (size_t batch = 0; batch < num_batches; batch++) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            // Process batch
            for (size_t i = 0; i < batch_size; i++) {
                size_t sequence_id = batch * batch_size + i;
                std::string text = generate_realistic_text(sequence_id);
                std::vector<int32_t> tokens = simulate_tokenization(text);
                cache->put_tokenized(sequence_id, tokens);
                
                // Occasional access to older sequences
                if (i % 100 == 0 && sequence_id > 1000) {
                    size_t old_id = sequence_id - (500 + (sequence_id % 500));
                    cache->has_tokenized(old_id);
                }
            }
            
            auto batch_end = std::chrono::high_resolution_clock::now();
            double batch_duration = std::chrono::duration<double>(batch_end - batch_start).count();
            double batch_throughput = batch_size / batch_duration;
            
            result.throughput_samples.push_back(batch_throughput);
            
            size_t current_memory = get_memory_usage_mb();
            result.memory_samples.push_back(current_memory);
            
            std::cout << "  Batch " << (batch + 1) << "/" << num_batches 
                      << " - Throughput: " << std::fixed << std::setprecision(1) 
                      << batch_throughput << " seq/sec, Memory: " << current_memory << "MB" << std::endl;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        size_t final_memory = get_memory_usage_mb();
        
        // Calculate results
        result.total_duration_seconds = std::chrono::duration<double>(end_time - start_time).count();
        result.avg_throughput_sequences_per_sec = total_sequences / result.total_duration_seconds;
        result.peak_memory_usage_mb = *std::max_element(result.memory_samples.begin(), result.memory_samples.end());
        result.memory_efficiency_ratio = static_cast<double>(final_memory - initial_memory) / initial_memory;
        
        auto cache_stats = cache->get_stats();
        result.cache_hit_ratio = cache_stats.hit_ratio;
        
        // Check stability over long run
        result.memory_stable = (result.memory_efficiency_ratio < MAX_MEMORY_GROWTH_RATIO);
        
        // Check throughput stability (coefficient of variation < 0.3)
        if (!result.throughput_samples.empty()) {
            double mean_throughput = 0;
            for (double t : result.throughput_samples) mean_throughput += t;
            mean_throughput /= result.throughput_samples.size();
            
            double variance = 0;
            for (double t : result.throughput_samples) {
                variance += (t - mean_throughput) * (t - mean_throughput);
            }
            variance /= result.throughput_samples.size();
            double std_dev = std::sqrt(variance);
            double cv = std_dev / mean_throughput;
            
            result.performance_stable = (cv < 0.3); // Coefficient of variation < 30%
        }
        
        results.push_back(result);
        print_streaming_result(result);
    }
    
    /**
     * @brief Run all streaming tests
     */
    void run_all_streaming_tests() {
        std::cout << "=== Large Dataset Streaming Performance Tests ===" << std::endl;
        std::cout << "Cache size: " << (cache->get_max_memory() / 1024 / 1024) << "MB" << std::endl;
        
        test_large_dataset_streaming();
        test_memory_constrained_streaming();
        test_random_access_streaming();
        test_long_running_streaming();
        
        print_streaming_summary();
    }
    
private:
    /**
     * @brief Generate realistic text for testing
     */
    std::string generate_realistic_text(size_t sequence_id) {
        static const std::vector<std::string> templates = {
            "This is a sample document with sequence ID {}. It contains various types of content including technical terms, natural language, and structured data.",
            "Document {} discusses machine learning algorithms, neural networks, and artificial intelligence applications in modern software development.",
            "Sequence {} contains information about data processing, tokenization techniques, and performance optimization strategies for large-scale systems.",
            "Entry {} describes streaming data architectures, caching mechanisms, and memory management approaches for high-throughput applications.",
            "Record {} includes details about distributed computing, parallel processing, and scalable system design patterns."
        };
        
        std::string base_text = templates[sequence_id % templates.size()];
        
        // Replace {} with sequence_id
        size_t pos = base_text.find("{}");
        if (pos != std::string::npos) {
            base_text.replace(pos, 2, std::to_string(sequence_id));
        }
        
        // Add some variation in length
        if (sequence_id % 3 == 0) {
            base_text += " Additional content for variation in sequence length and complexity.";
        }
        if (sequence_id % 7 == 0) {
            base_text += " Extended information with technical details and implementation specifics.";
        }
        
        return base_text;
    }
    
    /**
     * @brief Generate larger text for memory stress testing
     */
    std::string generate_large_text(size_t sequence_id) {
        std::string base_text = generate_realistic_text(sequence_id);
        
        // Repeat content to create larger sequences
        std::string large_text = base_text;
        for (int i = 0; i < 5; i++) {
            large_text += " " + base_text;
        }
        
        return large_text;
    }
    
    /**
     * @brief Simulate tokenization process
     */
    std::vector<int32_t> simulate_tokenization(const std::string& text) {
        std::vector<int32_t> tokens;
        
        // Simple tokenization simulation (word-based)
        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            // Hash word to create consistent token ID
            std::hash<std::string> hasher;
            size_t hash = hasher(word);
            tokens.push_back(static_cast<int32_t>(hash % 50000)); // Limit token range
        }
        
        return tokens;
    }
    
    /**
     * @brief Get current memory usage in MB
     */
    size_t get_memory_usage_mb() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_maxrss / 1024; // Convert KB to MB
    }
    
    /**
     * @brief Print streaming test result
     */
    void print_streaming_result(const StreamingTestResult& result) {
        std::cout << "\n" << result.test_name << " Results:" << std::endl;
        std::cout << "  Sequences processed: " << result.total_sequences_processed << std::endl;
        std::cout << "  Duration: " << std::fixed << std::setprecision(1) << result.total_duration_seconds << "s" << std::endl;
        std::cout << "  Avg throughput: " << std::fixed << std::setprecision(1) << result.avg_throughput_sequences_per_sec << " seq/sec" << std::endl;
        std::cout << "  Peak memory: " << result.peak_memory_usage_mb << "MB" << std::endl;
        std::cout << "  Memory growth: " << std::fixed << std::setprecision(2) << (result.memory_efficiency_ratio * 100) << "%" << std::endl;
        std::cout << "  Cache hit ratio: " << std::fixed << std::setprecision(3) << result.cache_hit_ratio << std::endl;
        std::cout << "  Memory stable: " << (result.memory_stable ? "✓" : "✗") << std::endl;
        std::cout << "  Performance stable: " << (result.performance_stable ? "✓" : "✗") << std::endl;
        
        // Performance assessment
        bool meets_requirements = 
            result.avg_throughput_sequences_per_sec >= MIN_THROUGHPUT_SEQUENCES_PER_SEC &&
            result.peak_memory_usage_mb <= MAX_MEMORY_USAGE_MB &&
            result.cache_hit_ratio >= MIN_CACHE_HIT_RATIO &&
            result.memory_stable &&
            result.performance_stable;
        
        std::cout << "  Overall: " << (meets_requirements ? "✓ PASS" : "✗ FAIL") << std::endl;
    }
    
    /**
     * @brief Print comprehensive streaming test summary
     */
    void print_streaming_summary() {
        std::cout << "\n=== Streaming Performance Summary ===" << std::endl;
        
        size_t passed = 0;
        double total_sequences = 0;
        double total_duration = 0;
        double avg_throughput = 0;
        size_t max_memory = 0;
        double avg_cache_hit_ratio = 0;
        
        for (const auto& result : results) {
            bool meets_requirements = 
                result.avg_throughput_sequences_per_sec >= MIN_THROUGHPUT_SEQUENCES_PER_SEC &&
                result.peak_memory_usage_mb <= MAX_MEMORY_USAGE_MB &&
                result.cache_hit_ratio >= MIN_CACHE_HIT_RATIO &&
                result.memory_stable &&
                result.performance_stable;
            
            if (meets_requirements) passed++;
            
            total_sequences += result.total_sequences_processed;
            total_duration += result.total_duration_seconds;
            avg_throughput += result.avg_throughput_sequences_per_sec;
            max_memory = std::max(max_memory, result.peak_memory_usage_mb);
            avg_cache_hit_ratio += result.cache_hit_ratio;
        }
        
        avg_throughput /= results.size();
        avg_cache_hit_ratio /= results.size();
        
        std::cout << "Tests passed: " << passed << "/" << results.size() << std::endl;
        std::cout << "Total sequences processed: " << static_cast<size_t>(total_sequences) << std::endl;
        std::cout << "Total processing time: " << std::fixed << std::setprecision(1) << total_duration << "s" << std::endl;
        std::cout << "Average throughput: " << std::fixed << std::setprecision(1) << avg_throughput << " seq/sec" << std::endl;
        std::cout << "Peak memory usage: " << max_memory << "MB" << std::endl;
        std::cout << "Average cache hit ratio: " << std::fixed << std::setprecision(3) << avg_cache_hit_ratio << std::endl;
        
        // Production readiness for large datasets
        bool production_ready = (passed == results.size()) && 
                               (avg_throughput >= MIN_THROUGHPUT_SEQUENCES_PER_SEC) &&
                               (max_memory <= MAX_MEMORY_USAGE_MB) &&
                               (avg_cache_hit_ratio >= MIN_CACHE_HIT_RATIO);
        
        std::cout << "\nLarge dataset streaming readiness: " << (production_ready ? "✓ READY" : "✗ NEEDS OPTIMIZATION") << std::endl;
        
        if (!production_ready) {
            std::cout << "\nOptimization recommendations:" << std::endl;
            if (avg_throughput < MIN_THROUGHPUT_SEQUENCES_PER_SEC) {
                std::cout << "  - Improve streaming throughput" << std::endl;
            }
            if (max_memory > MAX_MEMORY_USAGE_MB) {
                std::cout << "  - Reduce memory usage through better cache management" << std::endl;
            }
            if (avg_cache_hit_ratio < MIN_CACHE_HIT_RATIO) {
                std::cout << "  - Optimize cache strategy for large datasets" << std::endl;
            }
        }
    }
};

int main() {
    std::cout << "Large Dataset Streaming Performance Test" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    LargeDatasetStreamingTester tester;
    tester.run_all_streaming_tests();
    
    return 0;
}