#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <memory>
#include <cstring>
#include <functional>
#include <sys/resource.h>
#include <unistd.h>

// Include dataset headers
#include "common.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-text.h"
#include "llama-dataset.h"

struct StreamingOptimizationResult {
    std::string test_name;
    bool has_issues;
    double memory_efficiency_ratio;  // streaming/full memory usage
    double load_time_ratio;          // streaming/full load time
    bool data_consistency_ok;
    bool fallback_works;
    std::vector<std::string> issues_found;
    std::vector<std::string> optimizations_needed;
};

class StreamingOptimizer {
private:
    std::vector<StreamingOptimizationResult> results;

    size_t get_memory_usage() {
        struct rusage usage;
        getrusage(RUSAGE_SELF, &usage);
        return usage.ru_maxrss; // Peak memory in KB
    }

    double measure_time_ms(std::function<void()> func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

public:
    StreamingOptimizationResult analyze_gguf_streaming_optimization() {
        StreamingOptimizationResult result;
        result.test_name = "GGUF Streaming Optimization Analysis";
        result.has_issues = false;
        result.data_consistency_ok = true;
        result.fallback_works = true;

        std::cout << "\n=== GGUF Streaming Optimization Analysis ===" << std::endl;

        struct llama_dataset* full_dataset = nullptr;
        struct llama_dataset* streaming_dataset = nullptr;

        try {
            // Test with small dataset first
            std::string test_file = "test_data/small_dataset.gguf";

            // Measure full loading
            size_t memory_before_full = get_memory_usage();
            result.load_time_ratio = measure_time_ms([&]() {
                common_params params;
                params.in_files.push_back(test_file);
                full_dataset = llama_dataset_load_gguf(&params);
            });
            size_t memory_after_full = get_memory_usage();
            size_t full_memory = memory_after_full - memory_before_full;

            if (!full_dataset) {
                result.issues_found.push_back("Failed to load GGUF dataset in full mode");
                result.has_issues = true;
                return result;
            }

            // Measure streaming loading
            size_t memory_before_streaming = get_memory_usage();
            double streaming_load_time = measure_time_ms([&]() {
                common_params params;
                params.in_files.push_back(test_file);
                params.dataset_streaming = true;
                streaming_dataset = llama_dataset_load_gguf(&params);
            });
            size_t memory_after_streaming = get_memory_usage();
            size_t streaming_memory = memory_after_streaming - memory_before_streaming;

            if (!streaming_dataset) {
                result.issues_found.push_back("Failed to load GGUF dataset in streaming mode");
                result.has_issues = true;
                llama_dataset_free(full_dataset);
                return result;
            }

            // Calculate ratios
            result.load_time_ratio = streaming_load_time / std::max(result.load_time_ratio, 0.001);
            result.memory_efficiency_ratio = (double)streaming_memory / std::max((double)full_memory, 1.0);

            std::cout << "Load time ratio (streaming/full): " << result.load_time_ratio << std::endl;
            std::cout << "Memory ratio (streaming/full): " << result.memory_efficiency_ratio << std::endl;

            // Issue 1: Check if streaming actually provides memory benefits
            if (result.memory_efficiency_ratio >= 0.8) {
                result.issues_found.push_back("Streaming mode doesn't provide significant memory benefits");
                result.optimizations_needed.push_back("Optimize streaming to reduce initial memory allocation");
                result.has_issues = true;
            }

            // Issue 2: Check if streaming load time is reasonable
            if (result.load_time_ratio > 2.0) {
                result.issues_found.push_back("Streaming mode has significantly slower load time");
                result.optimizations_needed.push_back("Optimize streaming initialization to reduce overhead");
                result.has_issues = true;
            }

            // Issue 3: Test data consistency
            int full_count = llama_dataset_n_sequences(full_dataset);
            int streaming_count = llama_dataset_n_sequences(streaming_dataset);

            if (full_count != streaming_count) {
                result.issues_found.push_back("Sequence count mismatch between streaming and full modes");
                result.data_consistency_ok = false;
                result.has_issues = true;
            }

            // Issue 4: Test random access performance
            std::cout << "Testing random access performance..." << std::endl;

            auto test_random_access = [](struct llama_dataset* dataset) {
                auto start = std::chrono::high_resolution_clock::now();
                int seq_count = llama_dataset_n_sequences(dataset);
                for (int i = 0; i < std::min(50, seq_count); i++) {
                    int random_idx = (i * 7) % seq_count;
                    llama_dataset_sequence_length(dataset, random_idx);
                    llama_dataset_sequence(dataset, random_idx);
                }
                auto end = std::chrono::high_resolution_clock::now();
                return std::chrono::duration<double, std::milli>(end - start).count();
            };

            double full_random_time = test_random_access(full_dataset);
            double streaming_random_time = test_random_access(streaming_dataset);

            std::cout << "Random access time - Full: " << full_random_time << "ms, Streaming: " << streaming_random_time << "ms" << std::endl;

            if (streaming_random_time > full_random_time * 3.0) {
                result.issues_found.push_back("Streaming random access is significantly slower than full mode");
                result.optimizations_needed.push_back("Implement caching for frequently accessed sequences");
                result.has_issues = true;
            }

            // Issue 5: Test memory usage during random access
            size_t memory_before_access = get_memory_usage();
            test_random_access(streaming_dataset);
            size_t memory_after_access = get_memory_usage();
            size_t access_memory_growth = memory_after_access - memory_before_access;

            std::cout << "Memory growth during random access: " << access_memory_growth << "KB" << std::endl;

            if (access_memory_growth > full_memory * 0.5) {
                result.issues_found.push_back("Streaming mode accumulates too much memory during access");
                result.optimizations_needed.push_back("Implement LRU cache with memory limits for streaming data");
                result.has_issues = true;
            }

            // Issue 6: Test sequential access efficiency
            std::cout << "Testing sequential access efficiency..." << std::endl;

            auto test_sequential_access = [](struct llama_dataset* dataset) {
                auto start = std::chrono::high_resolution_clock::now();
                int seq_count = llama_dataset_n_sequences(dataset);
                for (int i = 0; i < std::min(20, seq_count); i++) {
                    llama_dataset_sequence_length(dataset, i);
                    llama_dataset_sequence(dataset, i);
                }
                auto end = std::chrono::high_resolution_clock::now();
                return std::chrono::duration<double, std::milli>(end - start).count();
            };

            double streaming_sequential_time = test_sequential_access(streaming_dataset);
            std::cout << "Sequential access time: " << streaming_sequential_time << "ms" << std::endl;

            if (streaming_sequential_time > 100.0) {  // Arbitrary threshold
                result.issues_found.push_back("Sequential access in streaming mode is slow");
                result.optimizations_needed.push_back("Implement read-ahead buffering for sequential access");
                result.has_issues = true;
            }

        } catch (const std::exception& e) {
            result.issues_found.push_back(std::string("Exception during analysis: ") + e.what());
            result.has_issues = true;
        }

        // Cleanup
        if (full_dataset) llama_dataset_free(full_dataset);
        if (streaming_dataset) llama_dataset_free(streaming_dataset);

        return result;
    }

    StreamingOptimizationResult analyze_parquet_streaming_optimization() {
        StreamingOptimizationResult result;
        result.test_name = "Parquet Streaming Optimization Analysis";
        result.has_issues = false;
        result.data_consistency_ok = true;
        result.fallback_works = true;

        std::cout << "\n=== Parquet Streaming Optimization Analysis ===" << std::endl;

        // Check if Parquet streaming is actually implemented
        bool parquet_streaming_supported = llama_dataset_supports_streaming(DATASET_PARQUET, "test_data/parquet_dataset.parquet");
        std::cout << "Parquet streaming supported: " << (parquet_streaming_supported ? "YES" : "NO") << std::endl;

        if (!parquet_streaming_supported) {
            result.issues_found.push_back("Parquet streaming is not properly implemented");
            result.optimizations_needed.push_back("Implement proper Parquet streaming with Arrow integration");
            result.has_issues = true;
            return result;
        }

        // Test Parquet streaming implementation
        common_params params;
        params.in_files.push_back("test_data/parquet_dataset.parquet");
        params.dataset_streaming = true;
#ifdef LLAMA_PARQUET
        struct llama_dataset* parquet_dataset = llama_dataset_from_parquet(&params);
        if (!parquet_dataset) {
            result.issues_found.push_back("Failed to load Parquet dataset in streaming mode");
            result.has_issues = true;
            return result;
        }

        bool is_streaming = llama_dataset_is_streaming_enabled(parquet_dataset);
        if (!is_streaming) {
            result.issues_found.push_back("Parquet dataset not actually in streaming mode");
            result.optimizations_needed.push_back("Fix Parquet streaming mode detection");
            result.has_issues = true;
        }

        llama_dataset_free(parquet_dataset);
#endif
        return result;
    }

    void analyze_streaming_fallback_mechanisms() {
        std::cout << "\n=== Streaming Fallback Mechanism Analysis ===" << std::endl;

        // Test text format fallback
        bool text_streaming_supported = llama_dataset_supports_streaming(DATASET_TEXT, "test_data/text_dataset.txt");
        std::cout << "Text streaming supported: " << (text_streaming_supported ? "YES" : "NO") << std::endl;

        if (text_streaming_supported) {
            std::cout << "⚠️  Text format incorrectly reports streaming support" << std::endl;
        } else {
            std::cout << "✅ Text format correctly reports no streaming support" << std::endl;
        }

        // Test loading text with streaming flag (should fallback)
        common_params params;
        params.in_files.push_back("test_data/text_dataset.txt");
        params.dataset_streaming = true;
        if (struct llama_dataset * text_dataset = llama_dataset_load_text_internal(&params, nullptr)) {
            if (llama_dataset_is_streaming_enabled(text_dataset)) {
                std::cout << "⚠️  Text dataset incorrectly enabled streaming mode" << std::endl;
            } else {
                std::cout << "✅ Text dataset correctly fell back to non-streaming mode" << std::endl;
            }
            llama_dataset_free(text_dataset);
        } else {
            std::cout << "ℹ️  Text dataset loading failed (expected due to null model)" << std::endl;
        }
    }

    void generate_optimization_report() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "STREAMING OPTIMIZATION ANALYSIS REPORT" << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        int total_issues = 0;
        for (const auto& result : results) {
            std::cout << "\n--- " << result.test_name << " ---" << std::endl;
            std::cout << "Status: " << (result.has_issues ? "NEEDS OPTIMIZATION" : "OPTIMAL") << std::endl;

            if (!result.issues_found.empty()) {
                std::cout << "\nIssues Found:" << std::endl;
                for (const auto& issue : result.issues_found) {
                    std::cout << "  ❌ " << issue << std::endl;
                    total_issues++;
                }
            }

            if (!result.optimizations_needed.empty()) {
                std::cout << "\nOptimizations Needed:" << std::endl;
                for (const auto& optimization : result.optimizations_needed) {
                    std::cout << "  🔧 " << optimization << std::endl;
                }
            }

            if (result.memory_efficiency_ratio > 0) {
                std::cout << "\nMemory Efficiency: " << (result.memory_efficiency_ratio < 0.8 ? "✅ GOOD" : "⚠️  NEEDS IMPROVEMENT") << std::endl;
                std::cout << "Memory Ratio: " << result.memory_efficiency_ratio << std::endl;
            }

            if (result.load_time_ratio > 0) {
                std::cout << "Load Time Efficiency: " << (result.load_time_ratio < 1.5 ? "✅ GOOD" : "⚠️  NEEDS IMPROVEMENT") << std::endl;
                std::cout << "Load Time Ratio: " << result.load_time_ratio << std::endl;
            }
        }

        std::cout << "\n--- Summary ---" << std::endl;
        std::cout << "Total Issues Found: " << total_issues << std::endl;

        if (total_issues == 0) {
            std::cout << "✅ Streaming implementation is well optimized!" << std::endl;
        } else {
            std::cout << "⚠️  Streaming implementation needs optimization" << std::endl;
        }

        // Generate specific optimization recommendations
        generate_optimization_recommendations();
    }

    void generate_optimization_recommendations() {
        std::cout << "\n--- Optimization Recommendations ---" << std::endl;

        std::cout << "\n1. Memory Management Optimizations:" << std::endl;
        std::cout << "   • Implement LRU cache for streaming data with configurable memory limits" << std::endl;
        std::cout << "   • Add memory pressure detection and automatic cache eviction" << std::endl;
        std::cout << "   • Optimize initial memory allocation in streaming mode" << std::endl;

        std::cout << "\n2. Performance Optimizations:" << std::endl;
        std::cout << "   • Implement read-ahead buffering for sequential access patterns" << std::endl;
        std::cout << "   • Add async I/O for background data loading" << std::endl;
        std::cout << "   • Optimize file seeking and reading operations" << std::endl;

        std::cout << "\n3. Fallback Mechanism Improvements:" << std::endl;
        std::cout << "   • Ensure proper fallback detection for unsupported formats" << std::endl;
        std::cout << "   • Add graceful degradation when streaming fails" << std::endl;
        std::cout << "   • Implement automatic retry with full loading mode" << std::endl;

        std::cout << "\n4. Data Consistency Improvements:" << std::endl;
        std::cout << "   • Add comprehensive data validation between streaming and full modes" << std::endl;
        std::cout << "   • Implement checksum verification for streaming data" << std::endl;
        std::cout << "   • Add error detection and recovery mechanisms" << std::endl;
    }

    void add_result(const StreamingOptimizationResult& result) {
        results.push_back(result);
    }
};

int main() {
    std::cout << "Streaming Optimization Analysis Tool" << std::endl;
    std::cout << "====================================" << std::endl;

    StreamingOptimizer optimizer;

    // Analyze GGUF streaming optimization
    StreamingOptimizationResult gguf_result = optimizer.analyze_gguf_streaming_optimization();
    optimizer.add_result(gguf_result);

    // Analyze Parquet streaming optimization
    StreamingOptimizationResult parquet_result = optimizer.analyze_parquet_streaming_optimization();
    optimizer.add_result(parquet_result);

    // Analyze fallback mechanisms
    optimizer.analyze_streaming_fallback_mechanisms();

    // Generate comprehensive optimization report
    optimizer.generate_optimization_report();

    return 0;
}
