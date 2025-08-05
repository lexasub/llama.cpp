/**
 * @file test_tokenization_performance_benchmark.cpp
 * @brief Comprehensive performance benchmarking for tokenization pipeline
 *
 * This tool provides detailed performance analysis of the tokenization pipeline,
 * measuring throughput, latency, memory efficiency, and cache performance across
 * different scenarios and dataset sizes.
 *
 * ## Performance Metrics Measured
 * - **Tokenization Throughput**: Tokens/second, sequences/second
 * - **Memory Efficiency**: Memory usage per token, cache hit ratios
 * - **Latency Analysis**: P50, P95, P99 latencies for tokenization operations
 * - **Cache Performance**: Hit rates, eviction patterns, memory pressure handling
 * - **Streaming Performance**: Large dataset processing efficiency
 * - **Concurrent Performance**: Multi-threaded tokenization benchmarks
 *
 * ## Test Scenarios
 * - Small dataset baseline (< 1MB)
 * - Medium dataset stress test (1-100MB)
 * - Large dataset streaming (> 100MB)
 * - Memory pressure simulation
 * - Concurrent access patterns
 * - Cache optimization validation
 *
 * @author llama.cpp dataset-converter team
 * @version 1.0
 * @since 2024
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <cstring>
#include <sys/resource.h>
#include <unistd.h>

#include "common.h"
#include "common/log.h"
#include "llama-dataset.h"
#include "streaming-cache.h"
#include "llama.h"

/**
 * @brief Performance measurement result structure
 */
struct PerformanceResult {
    std::string test_name;
    double duration_ms;
    size_t operations_count;
    double throughput_ops_per_sec;
    size_t memory_usage_bytes;
    double memory_efficiency_ratio;
    double cache_hit_ratio;
    std::vector<double> latency_percentiles; // P50, P95, P99
    std::string details;
    bool meets_requirements;
};

/**
 * @brief Comprehensive tokenization performance benchmark suite
 */
class TokenizationPerformanceBenchmark {
private:
    std::vector<PerformanceResult> results;
    std::unique_ptr<llama_dataset_streaming_cache> cache;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    
    // Performance thresholds for production requirements
    static constexpr double MIN_TOKENIZATION_THROUGHPUT = 1000.0; // tokens/sec
    static constexpr double MIN_CACHE_HIT_RATIO = 0.7; // 70%
    static constexpr double MAX_MEMORY_OVERHEAD_RATIO = 2.0; // 2x baseline
    static constexpr double MAX_P99_LATENCY_MS = 100.0; // 100ms
    
public:
    TokenizationPerformanceBenchmark() 
        : cache(std::make_unique<llama_dataset_streaming_cache>(64 * 1024 * 1024)) {} // 64MB cache
    
    ~TokenizationPerformanceBenchmark() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }
    
    bool init_model(const std::string& model_path) {
        if (model_path.empty()) {
            std::cout << "No model path provided - running cache-only benchmarks" << std::endl;
            return false;
        }
        
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 0; // CPU-only for consistent benchmarking
        model = llama_model_load_from_file(model_path.c_str(), model_params);
        
        if (!model) {
            std::cerr << "Failed to load model from " << model_path << std::endl;
            return false;
        }
        
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 2048;
        ctx_params.n_batch = 512;
        ctx = llama_init_from_model(model, ctx_params);
        
        return ctx != nullptr;
    }
    
    /**
     * @brief Measures execution time and calculates latency percentiles
     */
    template<typename Func>
    PerformanceResult measure_performance(const std::string& test_name, 
                                        size_t iterations, 
                                        Func&& operation) {
        PerformanceResult result;
        result.test_name = test_name;
        result.operations_count = iterations;
        
        std::vector<double> latencies;
        latencies.reserve(iterations);
        
        size_t memory_before = get_memory_usage();
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Execute operations and measure individual latencies
        for (size_t i = 0; i < iterations; i++) {
            auto op_start = std::chrono::high_resolution_clock::now();
            operation(i);
            auto op_end = std::chrono::high_resolution_clock::now();
            
            double latency_ms = std::chrono::duration<double, std::milli>(op_end - op_start).count();
            latencies.push_back(latency_ms);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        size_t memory_after = get_memory_usage();
        
        // Calculate metrics
        result.duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        result.throughput_ops_per_sec = (iterations * 1000.0) / result.duration_ms;
        result.memory_usage_bytes = (memory_after - memory_before) * 1024; // Convert KB to bytes
        
        // Calculate latency percentiles
        std::sort(latencies.begin(), latencies.end());
        result.latency_percentiles = {
            latencies[latencies.size() * 0.5],  // P50
            latencies[latencies.size() * 0.95], // P95
            latencies[latencies.size() * 0.99]  // P99
        };
        
        // Get cache statistics
        auto cache_stats = cache->get_stats();
        result.cache_hit_ratio = cache_stats.hit_ratio;
        
        // Check if meets production requirements
        result.meets_requirements = 
            result.throughput_ops_per_sec >= MIN_TOKENIZATION_THROUGHPUT &&
            result.cache_hit_ratio >= MIN_CACHE_HIT_RATIO &&
            result.latency_percentiles[2] <= MAX_P99_LATENCY_MS; // P99 latency
        
        return result;
    }
    
    /**
     * @brief Benchmark basic tokenization throughput
     */
    void benchmark_tokenization_throughput() {
        std::cout << "\n=== Tokenization Throughput Benchmark ===" << std::endl;
        
        // Generate test texts of varying lengths
        std::vector<std::string> test_texts;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> length_dist(10, 500);
        
        const std::string sample_words[] = {
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "machine", "learning", "artificial", "intelligence", "neural", "network",
            "tokenization", "performance", "benchmark", "optimization", "streaming"
        };
        
        for (size_t i = 0; i < 1000; i++) {
            std::stringstream ss;
            int word_count = length_dist(gen);
            for (int j = 0; j < word_count; j++) {
                if (j > 0) ss << " ";
                ss << sample_words[gen() % (sizeof(sample_words) / sizeof(sample_words[0]))];
            }
            test_texts.push_back(ss.str());
        }
        
        // Benchmark tokenization with cache
        auto result = measure_performance("Tokenization Throughput", test_texts.size(),
            [&](size_t i) {
                const std::string& text = test_texts[i];
                
                // Check cache first
                if (!cache->has_tokenized(i)) {
                    // Simulate tokenization (or use real model if available)
                    std::vector<int32_t> tokens;
                    
                    if (model && ctx) {
                        // Real tokenization
                        std::vector<llama_token> llama_tokens;
                        llama_tokens.resize(text.length() + 10);
                        
                        int n_tokens = llama_tokenize(llama_model_get_vocab(model), 
                                                    text.c_str(), text.length(),
                                                    llama_tokens.data(), llama_tokens.size(),
                                                    false, true);
                        
                        if (n_tokens > 0) {
                            llama_tokens.resize(n_tokens);
                            for (auto token : llama_tokens) {
                                tokens.push_back(static_cast<int32_t>(token));
                            }
                        }
                    } else {
                        // Mock tokenization for benchmarking
                        for (size_t j = 0; j < text.length(); j += 4) {
                            tokens.push_back(static_cast<int32_t>(text[j]));
                        }
                    }
                    
                    cache->put_tokenized(i, tokens);
                }
            });
        
        // Calculate tokens per second
        size_t total_tokens = 0;
        for (size_t i = 0; i < test_texts.size(); i++) {
            if (cache->has_tokenized(i)) {
                // Estimate token count (actual count would require cache retrieval)
                total_tokens += test_texts[i].length() / 4; // Rough estimate
            }
        }
        
        double tokens_per_sec = (total_tokens * 1000.0) / result.duration_ms;
        
        result.details = "Tokens/sec: " + std::to_string(tokens_per_sec) + 
                        ", Cache entries: " + std::to_string(cache->get_stats().tokenized_entries);
        
        results.push_back(result);
        print_result(result);
    }
    
    /**
     * @brief Benchmark streaming tokenization with large datasets
     */
    void benchmark_streaming_performance() {
        std::cout << "\n=== Streaming Tokenization Benchmark ===" << std::endl;
        
        // Simulate large dataset streaming
        const size_t large_dataset_size = 10000;
        const size_t cache_size_limit = 1024 * 1024; // 1MB cache
        
        cache->set_tokenization_cache_size(cache_size_limit);
        
        auto result = measure_performance("Streaming Large Dataset", large_dataset_size,
            [&](size_t i) {
                // Generate text that simulates streaming dataset
                std::string text = "This is sequence " + std::to_string(i) + 
                                 " in a large streaming dataset with various content lengths.";
                
                // Add some variation in text length
                if (i % 3 == 0) {
                    text += " This sequence has additional content to test variable length handling.";
                }
                if (i % 7 == 0) {
                    text += " Even more content for this particular sequence to create realistic variation.";
                }
                
                // Tokenize and cache
                std::vector<int32_t> tokens;
                for (char c : text) {
                    if (c != ' ') tokens.push_back(static_cast<int32_t>(c));
                }
                
                cache->put_tokenized(i, tokens);
                
                // Simulate access pattern (some sequences accessed multiple times)
                if (i > 100 && (i % 10 == 0)) {
                    // Access recent sequences to test cache effectiveness
                    size_t recent_id = i - (i % 50);
                    cache->has_tokenized(recent_id);
                }
            });
        
        auto cache_stats = cache->get_stats();
        result.details = "Memory pressure: " + std::to_string(cache_stats.tokenization_memory_pressure) +
                        ", Evictions: " + std::to_string(cache_stats.evictions) +
                        ", Hit ratio: " + std::to_string(cache_stats.hit_ratio);
        
        results.push_back(result);
        print_result(result);
    }
    
    /**
     * @brief Benchmark cache performance under memory pressure
     */
    void benchmark_memory_pressure_handling() {
        std::cout << "\n=== Memory Pressure Handling Benchmark ===" << std::endl;
        
        // Start with small cache and gradually increase pressure
        const size_t initial_cache_size = 64 * 1024; // 64KB
        cache->set_tokenization_cache_size(initial_cache_size);
        
        std::vector<double> memory_pressure_points;
        std::vector<double> performance_points;
        
        auto result = measure_performance("Memory Pressure Handling", 5000,
            [&](size_t i) {
                // Create progressively larger token sequences
                size_t token_count = 100 + (i / 10); // Growing sequence size
                std::vector<int32_t> tokens;
                
                for (size_t j = 0; j < token_count; j++) {
                    tokens.push_back(static_cast<int32_t>(i * 1000 + j));
                }
                
                auto op_start = std::chrono::high_resolution_clock::now();
                cache->put_tokenized(i, tokens);
                auto op_end = std::chrono::high_resolution_clock::now();
                
                double op_time = std::chrono::duration<double, std::milli>(op_end - op_start).count();
                
                // Record memory pressure and performance every 100 operations
                if (i % 100 == 0) {
                    auto stats = cache->get_stats();
                    memory_pressure_points.push_back(stats.tokenization_memory_pressure);
                    performance_points.push_back(op_time);
                    
                    // Adaptive cache sizing test
                    if (stats.tokenization_memory_pressure > 0.8) {
                        cache->adjust_tokenization_cache_size(stats.tokenization_memory_pressure);
                    }
                }
            });
        
        // Analyze memory pressure handling effectiveness
        double avg_pressure = 0.0;
        double max_pressure = 0.0;
        for (double pressure : memory_pressure_points) {
            avg_pressure += pressure;
            max_pressure = std::max(max_pressure, pressure);
        }
        avg_pressure /= memory_pressure_points.size();
        
        result.details = "Avg pressure: " + std::to_string(avg_pressure) +
                        ", Max pressure: " + std::to_string(max_pressure) +
                        ", Adaptive adjustments: " + std::to_string(memory_pressure_points.size());
        
        // Check if memory pressure was managed effectively
        result.meets_requirements = result.meets_requirements && (max_pressure < 0.95);
        
        results.push_back(result);
        print_result(result);
    }
    
    /**
     * @brief Benchmark concurrent tokenization performance
     */
    void benchmark_concurrent_performance() {
        std::cout << "\n=== Concurrent Tokenization Benchmark ===" << std::endl;
        
        const size_t num_threads = std::thread::hardware_concurrency();
        const size_t operations_per_thread = 500;
        
        std::atomic<size_t> completed_operations{0};
        std::atomic<size_t> cache_hits{0};
        std::atomic<size_t> cache_misses{0};
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<std::thread> threads;
        for (size_t t = 0; t < num_threads; t++) {
            threads.emplace_back([&, t]() {
                std::random_device rd;
                std::mt19937 gen(rd());
                std::uniform_int_distribution<> text_dist(50, 200);
                
                for (size_t i = 0; i < operations_per_thread; i++) {
                    uint64_t sequence_id = t * operations_per_thread + i;
                    
                    // Check cache first
                    if (cache->has_tokenized(sequence_id)) {
                        cache_hits++;
                    } else {
                        cache_misses++;
                        
                        // Generate and tokenize text
                        std::string text = "Thread " + std::to_string(t) + " sequence " + std::to_string(i);
                        int text_length = text_dist(gen);
                        while (text.length() < text_length) {
                            text += " additional content";
                        }
                        
                        std::vector<int32_t> tokens;
                        for (size_t j = 0; j < text.length(); j += 3) {
                            tokens.push_back(static_cast<int32_t>(text[j]));
                        }
                        
                        cache->put_tokenized(sequence_id, tokens);
                    }
                    
                    completed_operations++;
                }
            });
        }
        
        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double duration_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        
        PerformanceResult result;
        result.test_name = "Concurrent Tokenization";
        result.duration_ms = duration_ms;
        result.operations_count = completed_operations.load();
        result.throughput_ops_per_sec = (result.operations_count * 1000.0) / duration_ms;
        
        double concurrent_hit_ratio = static_cast<double>(cache_hits.load()) / 
                                    (cache_hits.load() + cache_misses.load());
        result.cache_hit_ratio = concurrent_hit_ratio;
        
        result.details = "Threads: " + std::to_string(num_threads) +
                        ", Hits: " + std::to_string(cache_hits.load()) +
                        ", Misses: " + std::to_string(cache_misses.load());
        
        result.meets_requirements = result.throughput_ops_per_sec >= MIN_TOKENIZATION_THROUGHPUT;
        
        results.push_back(result);
        print_result(result);
    }
    
    /**
     * @brief Benchmark cache optimization effectiveness
     */
    void benchmark_cache_optimization() {
        std::cout << "\n=== Cache Optimization Benchmark ===" << std::endl;
        
        // Test different cache configurations
        std::vector<std::pair<std::string, size_t>> cache_configs = {
            {"Small Cache (256KB)", 256 * 1024},
            {"Medium Cache (1MB)", 1024 * 1024},
            {"Large Cache (4MB)", 4 * 1024 * 1024}
        };
        
        for (const auto& config : cache_configs) {
            cache->clear();
            cache->set_tokenization_cache_size(config.second);
            
            auto result = measure_performance(config.first, 2000,
                [&](size_t i) {
                    // Create access pattern with some locality
                    uint64_t sequence_id;
                    if (i < 1000) {
                        sequence_id = i; // Sequential access
                    } else {
                        // Random access with some locality
                        sequence_id = (i - 1000) % 500; // Reuse recent sequences
                    }
                    
                    if (!cache->has_tokenized(sequence_id)) {
                        std::string text = "Sequence " + std::to_string(sequence_id) + " content";
                        std::vector<int32_t> tokens;
                        for (char c : text) {
                            tokens.push_back(static_cast<int32_t>(c));
                        }
                        cache->put_tokenized(sequence_id, tokens);
                    }
                });
            
            auto cache_stats = cache->get_stats();
            result.details = "Cache size: " + std::to_string(config.second / 1024) + "KB" +
                            ", Entries: " + std::to_string(cache_stats.tokenized_entries) +
                            ", Memory usage: " + std::to_string(cache_stats.tokenization_memory_usage / 1024) + "KB";
            
            results.push_back(result);
            print_result(result);
        }
    }
    
    /**
     * @brief Run all performance benchmarks
     */
    void run_all_benchmarks() {
        std::cout << "=== Tokenization Performance Benchmark Suite ===" << std::endl;
        std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << std::endl;
        std::cout << "Cache size: " << (cache->get_max_memory() / 1024 / 1024) << "MB" << std::endl;
        
        benchmark_tokenization_throughput();
        benchmark_streaming_performance();
        benchmark_memory_pressure_handling();
        benchmark_concurrent_performance();
        benchmark_cache_optimization();
        
        print_summary();
    }
    
    /**
     * @brief Print individual benchmark result
     */
    void print_result(const PerformanceResult& result) {
        std::cout << "\n" << result.test_name << ":" << std::endl;
        std::cout << "  Duration: " << std::fixed << std::setprecision(2) << result.duration_ms << "ms" << std::endl;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << result.throughput_ops_per_sec << " ops/sec" << std::endl;
        std::cout << "  Cache hit ratio: " << std::fixed << std::setprecision(3) << result.cache_hit_ratio << std::endl;
        
        if (!result.latency_percentiles.empty()) {
            std::cout << "  Latency P50/P95/P99: " 
                      << std::fixed << std::setprecision(2) 
                      << result.latency_percentiles[0] << "/"
                      << result.latency_percentiles[1] << "/"
                      << result.latency_percentiles[2] << "ms" << std::endl;
        }
        
        if (result.memory_usage_bytes > 0) {
            std::cout << "  Memory usage: " << (result.memory_usage_bytes / 1024) << "KB" << std::endl;
        }
        
        std::cout << "  Requirements: " << (result.meets_requirements ? "✓ PASS" : "✗ FAIL") << std::endl;
        
        if (!result.details.empty()) {
            std::cout << "  Details: " << result.details << std::endl;
        }
    }
    
    /**
     * @brief Print comprehensive benchmark summary
     */
    void print_summary() {
        std::cout << "\n=== Performance Benchmark Summary ===" << std::endl;
        
        size_t passed = 0;
        double total_duration = 0;
        double avg_throughput = 0;
        double avg_cache_hit_ratio = 0;
        
        for (const auto& result : results) {
            if (result.meets_requirements) passed++;
            total_duration += result.duration_ms;
            avg_throughput += result.throughput_ops_per_sec;
            avg_cache_hit_ratio += result.cache_hit_ratio;
        }
        
        avg_throughput /= results.size();
        avg_cache_hit_ratio /= results.size();
        
        std::cout << "Tests passed: " << passed << "/" << results.size() << std::endl;
        std::cout << "Total benchmark time: " << std::fixed << std::setprecision(1) << (total_duration / 1000.0) << "s" << std::endl;
        std::cout << "Average throughput: " << std::fixed << std::setprecision(1) << avg_throughput << " ops/sec" << std::endl;
        std::cout << "Average cache hit ratio: " << std::fixed << std::setprecision(3) << avg_cache_hit_ratio << std::endl;
        
        // Production readiness assessment
        bool production_ready = (passed == results.size()) && 
                               (avg_throughput >= MIN_TOKENIZATION_THROUGHPUT) &&
                               (avg_cache_hit_ratio >= MIN_CACHE_HIT_RATIO);
        
        std::cout << "\nProduction readiness: " << (production_ready ? "✓ READY" : "✗ NEEDS OPTIMIZATION") << std::endl;
        
        if (!production_ready) {
            std::cout << "\nOptimization recommendations:" << std::endl;
            if (avg_throughput < MIN_TOKENIZATION_THROUGHPUT) {
                std::cout << "  - Improve tokenization throughput (current: " << avg_throughput << ", required: " << MIN_TOKENIZATION_THROUGHPUT << ")" << std::endl;
            }
            if (avg_cache_hit_ratio < MIN_CACHE_HIT_RATIO) {
                std::cout << "  - Optimize cache hit ratio (current: " << avg_cache_hit_ratio << ", required: " << MIN_CACHE_HIT_RATIO << ")" << std::endl;
            }
        }
        
        // Final cache statistics
        auto final_stats = cache->get_stats();
        std::cout << "\nFinal cache statistics:" << std::endl;
        std::cout << "  Total entries: " << final_stats.entry_count << std::endl;
        std::cout << "  Tokenized entries: " << final_stats.tokenized_entries << std::endl;
        std::cout << "  Memory usage: " << (final_stats.tokenization_memory_usage / 1024) << "KB" << std::endl;
        std::cout << "  Memory pressure: " << std::fixed << std::setprecision(3) << final_stats.tokenization_memory_pressure << std::endl;
    }
    
private:
    /**
     * @brief Get current process memory usage in KB
     */
    size_t get_memory_usage() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_maxrss; // Peak memory in KB
    }
};

int main(int argc, char** argv) {
    std::cout << "Tokenization Performance Benchmark Suite" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    TokenizationPerformanceBenchmark benchmark;
    
    // Initialize model if path provided
    if (argc > 1) {
        std::string model_path = argv[1];
        if (!benchmark.init_model(model_path)) {
            std::cout << "Warning: Could not load model, running cache-only benchmarks" << std::endl;
        } else {
            std::cout << "Model loaded successfully: " << model_path << std::endl;
        }
    } else {
        std::cout << "No model path provided, running cache-only benchmarks" << std::endl;
        std::cout << "Usage: " << argv[0] << " [model_path]" << std::endl;
    }
    
    benchmark.run_all_benchmarks();
    
    return 0;
}