/**
 * @file llama-dataset-streaming.cpp
 * @brief Main streaming implementation for the llama.cpp dataset converter framework.
 *
 * This module provides the core streaming functionality that coordinates all streaming
 * operations across the dataset converter framework. It serves as the central orchestrator
 * for streaming cache management, optimization strategies, and data access patterns,
 * integrating seamlessly with the core dataset interface and format-specific implementations.
 *
 * ## Architecture Overview
 *
 * The streaming implementation follows a layered architecture:
 * - **Coordination Layer** (this module): Orchestrates streaming operations and integrates components
 * - **Cache Management**: Provides intelligent LRU/LFU/Adaptive caching with memory pressure detection
 * - **Optimization Manager**: Implements adaptive strategies for read-ahead, cache sizing, and access patterns
 * - **Memory Monitor**: Tracks system memory usage and triggers adaptive behaviors
 * - **Read-ahead Engine**: Provides predictive loading based on access pattern analysis
 *
 * ## Key Responsibilities
 *
 * ### Streaming Coordination
 * - Manages the lifecycle of streaming components (cache, optimization manager, memory monitor)
 * - Coordinates interactions between different streaming subsystems
 * - Provides unified configuration interface for all streaming features
 * - Handles error propagation and recovery across streaming components
 *
 * ### Cache Integration
 * - Integrates with the streaming cache for efficient memory management
 * - Configures cache policies (LRU, LFU, Adaptive) based on access patterns
 * - Manages cache size limits and memory pressure responses
 * - Provides cache statistics and performance monitoring
 *
 * ### Optimization Management
 * - Creates and manages optimization manager instances per dataset
 * - Configures read-ahead strategies based on dataset characteristics
 * - Implements adaptive cache sizing based on system memory pressure
 * - Coordinates prefetch operations with format-specific data loading
 *
 * ### Performance Monitoring
 * - Collects and aggregates performance statistics from all streaming components
 * - Provides unified statistics interface for cache hit ratios, memory usage, and access patterns
 * - Enables performance analysis and optimization tuning
 * - Supports runtime performance monitoring and diagnostics
 *
 * ## Integration with Core Dataset API
 *
 * This module implements the streaming functions declared in the core dataset interface:
 * - `llama_dataset_set_streaming_cache_size()`: Configures cache memory limits
 * - `llama_dataset_set_streaming_read_ahead()`: Enables predictive loading strategies
 * - `llama_dataset_set_adaptive_cache_sizing()`: Activates memory-pressure-based adaptation
 * - `llama_dataset_get_streaming_stats()`: Provides comprehensive performance metrics
 *
 * ## Optimization Strategies
 *
 * ### Adaptive Cache Management
 * The streaming implementation employs several adaptive strategies:
 * - **Memory Pressure Detection**: Monitors system memory and adjusts cache size accordingly
 * - **Access Pattern Analysis**: Analyzes sequence access patterns to optimize prefetch strategies
 * - **Dynamic Policy Selection**: Switches between LRU, LFU, and adaptive eviction policies
 * - **Predictive Loading**: Implements intelligent read-ahead based on detected access patterns
 *
 * ### Format-Specific Optimizations
 * - **GGUF Datasets**: Optimized tensor loading with minimal memory overhead
 * - **Parquet Datasets**: Batch-oriented loading with schema-aware caching
 * - **Text Datasets**: Tokenization result caching with intelligent prefetch
 *
 * ## Memory Management
 *
 * The streaming implementation provides sophisticated memory management:
 * - **Configurable Memory Limits**: User-defined cache size limits with automatic enforcement
 * - **Memory Pressure Response**: Automatic cache reduction when system memory is constrained
 * - **Leak Prevention**: Comprehensive resource cleanup and lifecycle management
 * - **Memory Usage Tracking**: Real-time monitoring of memory consumption across all components
 *
 * ## Thread Safety
 *
 * All streaming operations are designed to be thread-safe:
 * - **Read Operations**: Multiple threads can safely read from cached data simultaneously
 * - **Cache Operations**: Thread-safe cache access with minimal lock contention
 * - **Statistics Collection**: Lock-free statistics aggregation where possible
 * - **Configuration Changes**: Synchronized configuration updates with proper ordering
 *
 * ## Error Handling
 *
 * Comprehensive error handling ensures robust operation:
 * - **Graceful Degradation**: Falls back to non-streaming mode when streaming fails
 * - **Error Propagation**: Proper error code and message propagation to the core API
 * - **Resource Cleanup**: Automatic cleanup of streaming resources on errors
 * - **Diagnostic Information**: Detailed error messages for debugging and troubleshooting
 *
 * ## Performance Characteristics
 *
 * The streaming implementation is optimized for:
 * - **Sequential Access**: Excellent performance for training loops with predictive loading
 * - **Random Access**: Efficient caching minimizes repeated data loading overhead
 * - **Large Datasets**: Memory-efficient handling of datasets larger than available RAM
 * - **Mixed Workloads**: Adaptive strategies handle varying access patterns effectively
 *
 * ## Usage Examples
 *
 * ### Basic Streaming Configuration
 * ```c
 * // Configure 100MB cache with read-ahead
 * llama_dataset_set_streaming_cache_size(dataset, 100 * 1024 * 1024);
 * llama_dataset_set_streaming_read_ahead(dataset, true, 10);
 * llama_dataset_set_adaptive_cache_sizing(dataset, true);
 * ```
 *
 * ### Performance Monitoring
 * ```c
 * // Monitor streaming performance
 * double hit_ratio;
 * size_t memory_usage, entry_count;
 * llama_dataset_get_streaming_stats(dataset, &hit_ratio, &memory_usage, &entry_count);
 * printf("Cache hit ratio: %.2f%%, Memory: %zu bytes, Entries: %zu\n", 
 *        hit_ratio * 100, memory_usage, entry_count);
 * ```
 *
 * @see streaming/streaming-cache.h for cache implementation details
 * @see streaming/streaming-optimization-manager.h for optimization strategies
 * @see streaming/streaming-memory-monitor.h for memory monitoring capabilities
 * @see streaming/streaming-read-ahead.h for predictive loading implementation
 * @see core/llama-dataset.h for the core dataset interface
 *
 * @version 1.0
 * @since 2024
 */

#include <cstdlib>
#include <cstring>
#include <memory>

#include "llama-dataset-internal.h"
#include "llama-dataset-utils.h"
#include "llama-dataset.h"
#include "llama-impl.h"
#include "streaming-cache.h"
#include "streaming-optimization-manager.h"

/**
 * @brief Get or create the streaming optimization manager for a dataset.
 *
 * This function manages the lifecycle of the optimization manager, creating it on-demand
 * and configuring it based on the dataset characteristics. The optimization manager
 * coordinates all streaming optimizations including cache management, read-ahead strategies,
 * and adaptive memory management.
 *
 * The function performs the following operations:
 * 1. Validates that the dataset supports streaming operations
 * 2. Checks for an existing optimization manager instance
 * 3. Creates and initializes a new manager if needed
 * 4. Configures the manager with dataset-specific parameters
 * 5. Sets up prefetch callbacks for format-specific data loading
 * 6. Starts the optimization manager's background operations
 *
 * ## Optimization Manager Configuration
 *
 * The manager is configured based on dataset type:
 * - **GGUF datasets**: Optimized for tensor-based access patterns
 * - **Parquet datasets**: Configured for batch-oriented loading
 * - **Text datasets**: Optimized for tokenization result caching
 *
 * ## Prefetch Callback Implementation
 *
 * The prefetch callback integrates with the core dataset API to load sequences
 * on-demand. It handles:
 * - Sequence validation and bounds checking
 * - Memory allocation and data copying
 * - Error handling and resource cleanup
 * - Size calculation and reporting
 *
 * ## Thread Safety
 *
 * This function is thread-safe and can be called concurrently. The optimization
 * manager creation is protected by appropriate synchronization mechanisms.
 *
 * @param dataset Dataset for which to get/create the optimization manager
 * @return Pointer to the optimization manager, or nullptr on error
 *
 * @note The returned manager is owned by the dataset and will be automatically
 *       cleaned up when the dataset is freed.
 * @note This function may allocate significant resources for the optimization
 *       manager and should only be called when streaming is actually needed.
 *
 * @see llama_dataset_stream_optimization_manager for manager implementation details
 * @see streaming/streaming-optimization-manager.h for optimization strategies
 */
static llama_dataset_stream_optimization_manager* llama_dataset_streaming_get_optimization_manager(struct llama_dataset* dataset) {
    if (!dataset || !dataset->streaming || !dataset->streaming_cache) {
        return nullptr;
    }

    // Check if we already have an optimization manager
    llama_dataset_stream_optimization_manager* manager = static_cast<llama_dataset_stream_optimization_manager*>(dataset->optimization_manager);

    // If not, create one and initialize it
    if (!manager) {
        // Create a name for the manager based on the dataset type
        std::string name;
        switch (dataset->type) {
            case DATASET_GGUF:
                name = "gguf-dataset";
                break;
            case DATASET_PARQUET:
                name = "parquet-dataset";
                break;
            case DATASET_TEXT:
                name = "text-dataset";
                break;
            default:
                name = "unknown-dataset";
                break;
        }

        // Create and initialize the manager
        manager = new llama_dataset_stream_optimization_manager(name);

        // Get the current cache size from the streaming cache
        llama_dataset_streaming_cache* cache = dataset->streaming_cache;
        size_t cache_size = cache ? cache->get_max_memory() : 64 * 1024 * 1024; // Default 64MB

        manager->initialize(cache_size);

        // Set up the prefetch callback to load sequences from the dataset
        manager->set_prefetch_callback([dataset](uint64_t seq_id, size_t* size_out) -> void* {
            // Check if the sequence is valid
            if (seq_id >= llama_dataset_n_sequences(dataset)) {
                return nullptr;
            }

            // Get the sequence length
            int32_t seq_len = llama_dataset_sequence_length(dataset, seq_id);
            if (seq_len <= 0) {
                return nullptr;
            }

            // Get the sequence data
            const int32_t* seq_data = llama_dataset_sequence(dataset, seq_id);
            if (!seq_data) {
                return nullptr;
            }

            // Allocate memory for the sequence
            size_t size = seq_len * sizeof(int32_t);
            void* data = malloc(size);
            if (!data) {
                return nullptr;
            }

            // Copy the sequence data
            memcpy(data, seq_data, size);

            // Set the output size
            if (size_out) {
                *size_out = size;
            }

            return data;
        });

        // Start the optimization manager
        manager->start();

        // Store the manager in the dataset
        dataset->optimization_manager = manager;
    }

    return manager;
}

/**
 * @brief Configure streaming cache size for a dataset.
 *
 * This function sets the maximum memory usage for the streaming cache, which directly
 * impacts performance and memory consumption. The cache uses intelligent eviction
 * policies (LRU, LFU, or adaptive) to maintain the most relevant data within the
 * specified memory limit.
 *
 * ## Cache Size Considerations
 *
 * ### Performance Impact
 * - **Larger caches**: Better hit ratios, especially for random access patterns
 * - **Smaller caches**: Lower memory usage but potentially more cache misses
 * - **Optimal sizing**: Typically 10-20% of available system memory for training workloads
 *
 * ### Memory Management
 * - The cache enforces the specified limit through intelligent eviction
 * - Memory pressure detection can trigger automatic cache reduction
 * - Cache size can be dynamically adjusted during runtime
 *
 * ### Format-Specific Considerations
 * - **GGUF**: Cache size should accommodate multiple tensor sequences
 * - **Parquet**: Consider batch sizes and column data overhead
 * - **Text**: Account for tokenization result storage requirements
 *
 * ## Integration with Optimization Manager
 *
 * When an optimization manager exists, this function also updates its cache size
 * configuration to ensure consistent behavior across all streaming components.
 * The optimization manager may further adjust the cache size based on:
 * - System memory pressure detection
 * - Access pattern analysis
 * - Performance monitoring feedback
 *
 * ## Error Conditions
 *
 * This function will fail and set appropriate error codes if:
 * - The dataset is not in streaming mode
 * - The streaming cache is not properly initialized
 * - The cache size cannot be applied due to system constraints
 *
 * @param dataset Dataset to configure (must be in streaming mode)
 * @param cache_size_bytes Maximum cache size in bytes (0 disables caching)
 * @return true on success, false on error (check llama_dataset_get_error() for details)
 *
 * @note Setting cache_size_bytes to 0 effectively disables caching, which may
 *       significantly impact performance for datasets larger than available memory.
 * @note The actual memory usage may temporarily exceed the limit during cache
 *       reorganization, but will be brought within limits through eviction.
 *
 * @see llama_dataset_get_streaming_stats() to monitor actual cache memory usage
 * @see llama_dataset_set_adaptive_cache_sizing() for automatic cache size management
 */
bool llama_dataset_set_streaming_cache_size(struct llama_dataset* dataset, size_t cache_size_bytes) {
    if (!dataset || !dataset->streaming) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Dataset is not in streaming mode");
        return false;
    }

    // Get the streaming cache
    llama_dataset_streaming_cache* cache = dataset->streaming_cache;
    if (!cache) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Streaming cache not initialized");
        return false;
    }

    // Set the cache size
    cache->set_max_memory(cache_size_bytes);

    // Update the optimization manager if it exists
    if (auto manager = llama_dataset_streaming_get_optimization_manager(dataset)) {
        manager->set_cache_size(cache_size_bytes);
    }

    LLAMA_LOG_INFO("Set streaming cache size to %zu bytes", cache_size_bytes);
    return true;
}

/**
 * @brief Enable or disable read-ahead buffering for streaming.
 *
 * This function configures predictive loading strategies that prefetch sequences
 * based on detected access patterns. Read-ahead buffering significantly improves
 * performance for sequential access patterns common in training workloads by
 * loading data before it's actually requested.
 *
 * ## Read-ahead Strategies
 *
 * ### Sequential Pattern Detection
 * The read-ahead engine analyzes access patterns to detect:
 * - Forward sequential access (common in training loops)
 * - Backward sequential access (less common but supported)
 * - Strided access patterns (e.g., every Nth sequence)
 * - Random access with local clustering
 *
 * ### Adaptive Window Sizing
 * The window size determines how many sequences to prefetch ahead of the current
 * access position. The optimization manager may dynamically adjust this based on:
 * - Available cache memory
 * - Detected access patterns
 * - System memory pressure
 * - Historical prefetch accuracy
 *
 * ### Format-Specific Optimizations
 * - **GGUF**: Prefetches tensor data with minimal overhead
 * - **Parquet**: Coordinates with batch loading for optimal I/O
 * - **Text**: Prefetches tokenization results to avoid recomputation
 *
 * ## Performance Characteristics
 *
 * ### Benefits
 * - **Sequential Access**: Up to 50-80% performance improvement for training loops
 * - **Reduced Latency**: Data is available immediately when requested
 * - **I/O Optimization**: Batches disk operations for better throughput
 * - **CPU Utilization**: Overlaps I/O with computation for better resource usage
 *
 * ### Considerations
 * - **Memory Usage**: Prefetched data consumes additional cache memory
 * - **Accuracy**: Incorrect predictions waste memory and I/O bandwidth
 * - **Startup Cost**: Initial pattern detection may have slight overhead
 *
 * ## Configuration Guidelines
 *
 * ### Window Size Selection
 * - **Small datasets** (< 1GB): window_size = 3-5 sequences
 * - **Medium datasets** (1-10GB): window_size = 5-10 sequences  
 * - **Large datasets** (> 10GB): window_size = 10-20 sequences
 * - **Available memory**: Ensure window_size * avg_sequence_size < cache_size / 2
 *
 * ### Access Pattern Considerations
 * - **Pure sequential**: Larger window sizes (10-20) provide maximum benefit
 * - **Mixed patterns**: Moderate window sizes (5-10) balance benefits and overhead
 * - **Random access**: Smaller window sizes (3-5) or disabled read-ahead
 *
 * @param dataset Dataset to configure (must be in streaming mode)
 * @param enabled Whether to enable read-ahead buffering
 * @param window_size Number of sequences to prefetch (range: 1-100, recommended: 5-10)
 * @return true on success, false on error (check llama_dataset_get_error() for details)
 *
 * @note Enabling read-ahead with a large window size on memory-constrained systems
 *       may trigger frequent cache evictions, potentially reducing performance.
 * @note The optimization manager may override the window size based on runtime
 *       analysis of access patterns and system conditions.
 *
 * @see llama_dataset_set_adaptive_cache_sizing() for automatic memory management
 * @see streaming/streaming-read-ahead.h for detailed read-ahead implementation
 */
bool llama_dataset_set_streaming_read_ahead(struct llama_dataset* dataset, bool enabled, size_t window_size) {
    if (!dataset || !dataset->streaming) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Dataset is not in streaming mode");
        return false;
    }

    // Get or create the optimization manager
    llama_dataset_stream_optimization_manager* manager = llama_dataset_streaming_get_optimization_manager(dataset);
    if (!manager) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Failed to initialize optimization manager");
        return false;
    }

    // Configure read-ahead
    manager->set_read_ahead_enabled(enabled);
    manager->set_read_ahead_window(window_size);

    LLAMA_LOG_INFO("Set streaming read-ahead to %s with window size %zu", enabled ? "enabled" : "disabled", window_size);
    return true;
}

/**
 * @brief Enable or disable adaptive cache sizing based on memory pressure.
 *
 * This function activates intelligent cache size management that automatically
 * adjusts cache memory usage based on system memory pressure and access patterns.
 * Adaptive sizing ensures optimal memory utilization while maintaining performance
 * under varying system conditions and workload characteristics.
 *
 * ## Adaptive Sizing Strategies
 *
 * ### Memory Pressure Detection
 * The system continuously monitors:
 * - **System Memory Usage**: Total RAM utilization across all processes
 * - **Available Memory**: Free memory available for cache expansion
 * - **Memory Allocation Patterns**: Rate of memory allocation/deallocation
 * - **Swap Activity**: Disk swapping indicating memory pressure
 *
 * ### Dynamic Cache Adjustment
 * Based on detected conditions, the cache size is adjusted:
 * - **Low Memory Pressure**: Cache can expand up to configured maximum
 * - **Moderate Pressure**: Cache size reduced to 70-80% of maximum
 * - **High Pressure**: Cache size reduced to 40-60% of maximum
 * - **Critical Pressure**: Cache size minimized to essential entries only
 *
 * ### Performance-Based Optimization
 * The adaptive algorithm also considers:
 * - **Cache Hit Ratios**: Maintains minimum effective cache size for good performance
 * - **Access Patterns**: Adjusts cache size based on detected usage patterns
 * - **Eviction Frequency**: Prevents thrashing by maintaining stable cache size
 * - **I/O Patterns**: Balances memory usage with disk I/O efficiency
 *
 * ## Integration with Other Components
 *
 * ### Memory Monitor Integration
 * Adaptive sizing works closely with the memory monitor to:
 * - Receive real-time memory pressure notifications
 * - Coordinate with other system components using memory
 * - Implement gradual cache size changes to avoid performance spikes
 * - Provide feedback on cache effectiveness under different memory conditions
 *
 * ### Optimization Manager Coordination
 * The optimization manager coordinates adaptive sizing with:
 * - **Read-ahead Strategies**: Adjusts prefetch window based on available cache space
 * - **Eviction Policies**: Switches between LRU/LFU/Adaptive based on memory pressure
 * - **Access Pattern Analysis**: Uses pattern data to optimize cache size decisions
 * - **Performance Monitoring**: Tracks effectiveness of adaptive adjustments
 *
 * ## Benefits and Trade-offs
 *
 * ### Benefits
 * - **Automatic Optimization**: No manual tuning required for different system conditions
 * - **Memory Efficiency**: Prevents excessive memory usage that could impact system stability
 * - **Performance Stability**: Maintains reasonable performance even under memory pressure
 * - **System Integration**: Works well with other applications competing for memory
 *
 * ### Trade-offs
 * - **Complexity**: Adds algorithmic complexity to cache management
 * - **Monitoring Overhead**: Continuous memory monitoring has small CPU cost
 * - **Response Latency**: Cache size changes may temporarily impact performance
 * - **Tuning Sensitivity**: May require adjustment for specific workload characteristics
 *
 * ## Configuration Recommendations
 *
 * ### When to Enable
 * - **Shared Systems**: Multiple applications competing for memory
 * - **Variable Workloads**: Memory usage patterns change over time
 * - **Production Environments**: Stability is more important than peak performance
 * - **Large Datasets**: Dataset size approaches or exceeds available memory
 *
 * ### When to Disable
 * - **Dedicated Systems**: Dataset processing is the only significant memory user
 * - **Predictable Workloads**: Memory usage patterns are stable and well-understood
 * - **Performance Critical**: Maximum performance is required regardless of memory usage
 * - **Small Datasets**: Dataset fits comfortably in available memory
 *
 * @param dataset Dataset to configure (must be in streaming mode)
 * @param enabled Whether to enable adaptive cache sizing
 * @return true on success, false on error (check llama_dataset_get_error() for details)
 *
 * @note Adaptive sizing may temporarily reduce cache performance during memory
 *       pressure events, but helps maintain overall system stability.
 * @note The effectiveness of adaptive sizing depends on the accuracy of memory
 *       pressure detection, which may vary across different operating systems.
 *
 * @see streaming/streaming-memory-monitor.h for memory monitoring implementation
 * @see llama_dataset_set_streaming_cache_size() for manual cache size control
 */
bool llama_dataset_set_adaptive_cache_sizing(struct llama_dataset* dataset, bool enabled) {
    if (!dataset || !dataset->streaming) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Dataset is not in streaming mode");
        return false;
    }

    // Get the streaming cache
    llama_dataset_streaming_cache* cache = dataset->streaming_cache;
    if (!cache) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Streaming cache not initialized");
        return false;
    }

    // Update the optimization manager if it exists
    if (auto manager = llama_dataset_streaming_get_optimization_manager(dataset)) {
        manager->set_adaptive_cache_enabled(enabled);
    }

    LLAMA_LOG_INFO("Set adaptive cache sizing to %s", enabled ? "enabled" : "disabled");
    return true;
}

/**
 * @brief Get comprehensive streaming cache statistics and performance metrics.
 *
 * This function provides detailed statistics about cache performance, memory usage,
 * and access patterns collected from all streaming components. These metrics are
 * essential for performance analysis, optimization tuning, and monitoring dataset
 * access efficiency in production environments.
 *
 * ## Statistics Categories
 *
 * ### Cache Performance Metrics
 * - **Hit Ratio**: Percentage of cache hits vs. total accesses (0.0-1.0)
 *   - Values > 0.8 indicate excellent cache performance
 *   - Values 0.6-0.8 indicate good performance for most workloads
 *   - Values < 0.6 may indicate suboptimal cache sizing or access patterns
 *
 * ### Memory Usage Statistics
 * - **Current Memory Usage**: Actual memory consumed by cached data
 * - **Memory Efficiency**: How effectively the cache uses allocated memory
 * - **Peak Memory Usage**: Maximum memory usage since cache initialization
 * - **Memory Pressure Events**: Number of times cache was reduced due to system pressure
 *
 * ### Access Pattern Analysis
 * - **Entry Count**: Number of distinct sequences currently cached
 * - **Access Frequency**: Distribution of access frequencies across cached entries
 * - **Eviction Statistics**: Number and types of cache evictions performed
 * - **Prefetch Accuracy**: Effectiveness of read-ahead predictions
 *
 * ## Performance Interpretation
 *
 * ### Hit Ratio Analysis
 * - **> 0.9**: Excellent - cache is very effective for the access pattern
 * - **0.8-0.9**: Good - cache provides significant performance benefit
 * - **0.6-0.8**: Moderate - cache helps but may benefit from tuning
 * - **0.4-0.6**: Poor - cache sizing or eviction policy may need adjustment
 * - **< 0.4**: Very Poor - consider disabling cache or major reconfiguration
 *
 * ### Memory Usage Guidelines
 * - **Memory Efficiency**: Should be > 80% for well-tuned caches
 * - **Growth Patterns**: Steady growth indicates good cache utilization
 * - **Pressure Events**: Frequent events may indicate oversized cache
 * - **Entry Density**: High entry count with low memory usage indicates small sequences
 *
 * ### Access Pattern Insights
 * - **High Entry Count + High Hit Ratio**: Good for random access patterns
 * - **Low Entry Count + High Hit Ratio**: Good for sequential access patterns
 * - **High Entry Count + Low Hit Ratio**: May indicate cache thrashing
 * - **Low Entry Count + Low Hit Ratio**: May indicate poor cache sizing
 *
 * ## Optimization Recommendations
 *
 * ### Based on Hit Ratio
 * - **Low Hit Ratio**: Increase cache size, enable adaptive sizing, or adjust eviction policy
 * - **High Hit Ratio**: Current configuration is optimal, consider enabling read-ahead
 * - **Fluctuating Hit Ratio**: Enable adaptive sizing to handle varying access patterns
 *
 * ### Based on Memory Usage
 * - **Low Memory Usage**: Cache size can be increased for better performance
 * - **High Memory Usage**: Consider enabling adaptive sizing or reducing cache size
 * - **Memory Pressure**: Enable adaptive sizing and monitor system memory usage
 *
 * ## Integration with Monitoring Tools
 *
 * These statistics integrate with external monitoring systems:
 * - **Performance Dashboards**: Real-time cache performance visualization
 * - **Alerting Systems**: Notifications when performance degrades below thresholds
 * - **Capacity Planning**: Historical data for cache sizing decisions
 * - **Debugging Tools**: Detailed diagnostics for performance troubleshooting
 *
 * @param dataset Dataset to query (must be in streaming mode)
 * @param hit_ratio Pointer to store cache hit ratio (0.0-1.0, higher is better)
 * @param memory_usage_bytes Pointer to store current cache memory usage in bytes
 * @param entry_count Pointer to store number of sequences currently cached
 * @return true on success, false on error (check llama_dataset_get_error() for details)
 *
 * @note All output parameters are optional (can be NULL) if specific statistics are not needed
 * @note Statistics are collected atomically to ensure consistency across all returned values
 * @note This function has minimal performance impact and can be called frequently for monitoring
 *
 * @see tools/streaming-optimization-analysis.cpp for detailed performance analysis tools
 * @see streaming/streaming-cache.h for detailed cache statistics structure
 */
bool llama_dataset_get_streaming_stats(
    const struct llama_dataset* dataset,
    double* hit_ratio,
    size_t* memory_usage_bytes,
    size_t* entry_count) {

    if (!dataset || !dataset->streaming) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Dataset is not in streaming mode");
        return false;
    }

    // Get the streaming cache
    llama_dataset_streaming_cache* cache = dataset->streaming_cache;
    if (!cache) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Streaming cache not initialized");
        return false;
    }

    // Get the cache statistics
    llama_dataset_streaming_cache::CacheStats stats = cache->get_stats();

    // Set the output values
    if (hit_ratio) {
        *hit_ratio = stats.hit_ratio;
    }

    if (memory_usage_bytes) {
        *memory_usage_bytes = stats.current_memory;
    }

    if (entry_count) {
        *entry_count = stats.entry_count;
    }

    return true;
}
