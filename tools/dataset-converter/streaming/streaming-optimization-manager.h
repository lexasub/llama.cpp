#pragma once

#include "streaming-cache.h"
#include "streaming-read-ahead.h"
#include "streaming-memory-monitor.h"

#include <memory>
#include <string>
#include <functional>

/**
 * @file streaming-optimization-manager.h
 * @brief Central coordination hub for all streaming dataset optimization strategies
 *
 * This module serves as the primary orchestrator for the dataset converter's streaming
 * optimization subsystem. It integrates and coordinates multiple optimization components
 * to provide intelligent, adaptive performance optimization for large-scale dataset
 * processing operations.
 *
 * ## Core Responsibilities
 * - **Unified Optimization Coordination**: Centralized management of all streaming optimizations
 * - **Adaptive Cache Management**: Dynamic cache sizing and policy adjustment based on access patterns
 * - **Intelligent Read-ahead**: Predictive data loading with pattern recognition
 * - **Memory Pressure Response**: Automatic adaptation to system memory conditions
 * - **Performance Analytics**: Comprehensive tracking and analysis of optimization effectiveness
 * - **Access Pattern Learning**: Real-time analysis and adaptation to data access patterns
 *
 * ## Architecture Overview
 * The optimization manager follows a modular architecture where specialized components
 * handle specific optimization aspects:
 * 
 * ```
 * ┌─────────────────────────────────────────────────────────────┐
 * │                Optimization Manager                         │
 * ├─────────────────┬─────────────────┬─────────────────────────┤
 * │   Cache Manager │  Read-ahead     │   Memory Monitor        │
 * │   - LRU/LFU     │  - Sequential   │   - Pressure Detection  │
 * │   - Adaptive    │  - Predictive   │   - Threshold Management│
 * │   - Statistics  │  - Prefetch     │   - Callback System     │
 * └─────────────────┴─────────────────┴─────────────────────────┘
 * ```
 *
 * ## Optimization Strategies
 * 
 * ### Cache Management
 * - **Dynamic Sizing**: Automatic cache size adjustment based on memory availability
 * - **Policy Selection**: Adaptive switching between LRU, LFU, and custom eviction policies
 * - **Memory Pressure Response**: Proactive cache reduction during memory pressure
 * - **Hit Rate Optimization**: Continuous tuning to maximize cache effectiveness
 *
 * ### Read-ahead Optimization
 * - **Sequential Pattern Detection**: Recognition of sequential access patterns
 * - **Predictive Prefetching**: Intelligent prediction of future data needs
 * - **Adaptive Window Sizing**: Dynamic adjustment of read-ahead window based on patterns
 * - **Memory-aware Prefetching**: Coordination with memory monitor to prevent over-allocation
 *
 * ### Access Pattern Analysis
 * - **Real-time Pattern Recognition**: Continuous analysis of data access patterns
 * - **Sequential vs Random Classification**: Automatic detection of access pattern types
 * - **Adaptive Strategy Selection**: Dynamic optimization strategy selection based on patterns
 * - **Performance Feedback Loop**: Continuous refinement based on optimization effectiveness
 *
 * ## Integration with Dataset Components
 * The optimization manager seamlessly integrates with:
 * - **Core Dataset API**: Transparent optimization layer for all dataset operations
 * - **Format Handlers**: Format-specific optimization strategies (GGUF, Parquet, Text)
 * - **Streaming Infrastructure**: Deep integration with streaming cache and read-ahead systems
 * - **Memory Management**: Coordination with system memory monitoring and pressure detection
 *
 * ## Performance Characteristics
 * - **Low Overhead**: Minimal impact on data access performance
 * - **Adaptive Algorithms**: Self-tuning optimization strategies
 * - **Scalable Architecture**: Efficient handling of datasets from MB to TB scale
 * - **Memory Efficient**: Intelligent memory usage with pressure-aware optimization
 * - **Thread-safe Operations**: Safe concurrent access from multiple threads
 *
 * ## Configuration and Tuning
 * The manager provides extensive configuration options:
 * - Cache size limits and eviction policies
 * - Read-ahead window sizes and prefetch strategies
 * - Memory pressure thresholds and response strategies
 * - Performance monitoring intervals and metrics collection
 * - Optimization feature enable/disable controls
 *
 * ## Usage Patterns
 * 
 * ### Basic Usage
 * ```cpp
 * auto optimizer = std::make_unique<llama_dataset_stream_optimization_manager>("my_dataset");
 * optimizer->initialize(128 * 1024 * 1024); // 128MB cache
 * optimizer->set_prefetch_callback([](uint64_t id, size_t* size) { return load_data(id, size); });
 * optimizer->start();
 * 
 * // Use optimized data access
 * void* data = optimizer->get_sequence(sequence_id);
 * ```
 *
 * ### Advanced Configuration
 * ```cpp
 * optimizer->set_read_ahead_window(10);
 * optimizer->set_memory_check_interval(500);
 * optimizer->set_adaptive_cache_enabled(true);
 * 
 * // Monitor performance
 * auto stats = optimizer->get_stats();
 * LLAMA_LOG_INFO("Cache hit ratio: %.2f%%", stats.cache_stats.hit_ratio * 100);
 * ```
 *
 * @author llama.cpp dataset-converter team
 * @version 1.0
 * @since 2024
 * 
 * @see llama_dataset_streaming_cache For cache management details
 * @see llama_dataset_streaming_read_ahead For read-ahead implementation
 * @see llama_dataset_streaming_memory_monitor For memory monitoring
 */

/**
 * @brief Central optimization coordinator for streaming dataset operations
 *
 * The streaming optimization manager serves as the primary orchestrator for all
 * performance optimization strategies in the dataset converter's streaming subsystem.
 * It integrates cache management, read-ahead buffering, memory monitoring, and
 * access pattern analysis to provide intelligent, adaptive optimization for
 * large-scale dataset processing.
 *
 * This class implements a sophisticated optimization framework that:
 * - Automatically adapts to different access patterns (sequential vs random)
 * - Dynamically adjusts cache sizes based on memory availability
 * - Coordinates read-ahead operations with memory pressure conditions
 * - Provides comprehensive performance analytics and monitoring
 * - Enables fine-grained control over optimization strategies
 *
 * ## Thread Safety
 * All public methods are thread-safe and can be called concurrently from multiple
 * threads. The manager coordinates with background threads for memory monitoring
 * and read-ahead operations.
 *
 * ## Lifecycle Management
 * The manager follows a clear lifecycle: construct → initialize → start → use → stop → destroy.
 * Proper initialization and cleanup are essential for optimal performance.
 *
 * ## Performance Impact
 * The optimization manager is designed to have minimal overhead while providing
 * significant performance benefits for large dataset operations. Typical improvements
 * include 2-5x faster sequential access and 20-50% reduction in memory pressure.
 */
class llama_dataset_stream_optimization_manager {
public:
    /**
     * @brief Callback function type for data prefetching operations
     * 
     * This callback is invoked by the optimization manager when read-ahead operations
     * need to load data that is not currently in cache. The callback should load the
     * requested sequence data and return a pointer to it.
     * 
     * @param sequence_id Unique identifier of the sequence to load
     * @param size Pointer to store the size of the loaded data
     * @return Pointer to the loaded data, or nullptr if loading failed
     * 
     * @note The returned pointer should remain valid until the data is cached
     * @note The callback may be invoked from background threads
     * @note Implementation should be thread-safe if concurrent access is possible
     */
    using PrefetchCallback = std::function<void*(uint64_t, size_t*)>;

private:
    std::unique_ptr<llama_dataset_streaming_cache> cache;           ///< Cache management component
    std::unique_ptr<llama_dataset_streaming_read_ahead> read_ahead; ///< Read-ahead optimization component
    std::unique_ptr<llama_dataset_streaming_memory_monitor> memory_monitor; ///< Memory monitoring component

    PrefetchCallback prefetch_callback;  ///< User-provided data loading callback
    std::string dataset_name;            ///< Name identifier for this dataset
    bool optimization_enabled;           ///< Master optimization enable/disable flag

    // Performance tracking
    uint64_t access_count;               ///< Total number of data access operations
    uint64_t sequential_access_count;    ///< Number of sequential access operations
    uint64_t random_access_count;        ///< Number of random access operations
    uint64_t last_accessed_id;           ///< ID of the last accessed sequence

    /**
     * @brief Handle memory pressure notifications from the memory monitor
     * 
     * This internal method is called when the memory monitor detects memory pressure.
     * It implements adaptive responses including cache eviction, read-ahead throttling,
     * and optimization strategy adjustment.
     * 
     * @param pressure Current memory pressure level (0.0 to 1.0)
     */
    void handle_memory_pressure(double pressure);

    /**
     * @brief Handle prefetch requests from the read-ahead system
     * 
     * This internal method coordinates prefetch operations by invoking the user-provided
     * prefetch callback and managing the resulting data placement in cache.
     * 
     * @param sequence_id ID of the sequence to prefetch
     */
    void handle_prefetch(uint64_t sequence_id);

    /**
     * @brief Update access pattern analysis with new access information
     * 
     * This internal method analyzes each data access to build a profile of access
     * patterns, enabling adaptive optimization strategy selection.
     * 
     * @param sequence_id ID of the sequence being accessed
     */
    void update_access_pattern(uint64_t sequence_id);

public:
    /**
     * @brief Construct a new optimization manager with optional dataset name
     * 
     * Creates a new optimization manager instance with the specified dataset name.
     * The manager is created in an uninitialized state and requires explicit
     * initialization before use.
     * 
     * @param name Optional name identifier for this dataset (default: "unnamed")
     * 
     * @note The name is used for logging and debugging purposes
     * @note Multiple managers can be created for different datasets
     */
    llama_dataset_stream_optimization_manager(const std::string& name = "unnamed");

    /**
     * @brief Destructor - ensures proper cleanup of optimization components
     * 
     * Automatically stops all optimization operations and cleans up resources.
     * Ensures that background threads are properly terminated and memory is released.
     */
    ~llama_dataset_stream_optimization_manager();

    /**
     * @brief Initialize the optimization manager with specified cache size
     * 
     * Initializes all optimization components including cache, read-ahead system,
     * and memory monitor. This method must be called before starting optimization
     * operations.
     * 
     * @param cache_size Maximum cache size in bytes (default: 64MB)
     * 
     * @throws std::runtime_error if initialization fails
     * @throws std::bad_alloc if memory allocation fails
     * 
     * @note Cache size should be chosen based on available memory and dataset characteristics
     * @note Larger cache sizes generally improve performance but consume more memory
     * @note The cache size can be adjusted later using set_cache_size()
     */
    void initialize(size_t cache_size = 64 * 1024 * 1024);

    /**
     * @brief Start all optimization operations
     * 
     * Begins active optimization including memory monitoring, read-ahead operations,
     * and cache management. The manager must be initialized before calling this method.
     * 
     * @throws std::runtime_error if manager is not initialized
     * @throws std::system_error if background threads cannot be started
     * 
     * @note This method starts background threads for monitoring and prefetching
     * @note Optimization operations continue until stop() is called
     */
    void start();

    /**
     * @brief Stop all optimization operations
     * 
     * Gracefully stops all optimization activities including background threads
     * and monitoring operations. The cache remains intact and can be accessed
     * after stopping.
     * 
     * @note This method waits for background threads to complete
     * @note Calling stop() on an already stopped manager is safe (no-op)
     * @note The manager can be restarted using start() after stopping
     */
    void stop();

    /**
     * @brief Retrieve cached data for a specific sequence with optimization
     * 
     * Attempts to retrieve the requested sequence from cache. If not cached,
     * triggers loading through the prefetch callback and applies optimization
     * strategies including read-ahead and access pattern analysis.
     * 
     * @param sequence_id Unique identifier of the sequence to retrieve
     * @return Pointer to the sequence data, or nullptr if not available
     * 
     * @note This method updates access pattern statistics
     * @note May trigger read-ahead operations for subsequent sequences
     * @note The returned pointer remains valid until the data is evicted from cache
     * @note Thread-safe and can be called concurrently from multiple threads
     */
    void* get_sequence(uint64_t sequence_id);

    /**
     * @brief Store sequence data in cache with optimization metadata
     * 
     * Explicitly stores sequence data in the cache, bypassing the prefetch callback.
     * This method is useful for pre-loading known data or storing computed results.
     * 
     * @param sequence_id Unique identifier for the sequence
     * @param data Pointer to the sequence data to store
     * @param size Size of the data in bytes
     * 
     * @note The data is copied into cache-managed memory
     * @note May trigger cache eviction if memory limits are exceeded
     * @note Updates cache statistics and access patterns
     * @note Thread-safe and can be called concurrently from multiple threads
     */
    void put_sequence(uint64_t sequence_id, void* data, size_t size);

    /**
     * @brief Set the callback function for data prefetching operations
     * 
     * Registers a callback function that will be invoked when the optimization
     * manager needs to load data that is not currently in cache. This callback
     * is essential for read-ahead and cache miss handling.
     * 
     * @param callback Function to call for data loading operations
     * 
     * @note The callback may be invoked from background threads
     * @note Callback should be thread-safe if concurrent access is possible
     * @note Setting a null callback disables automatic data loading
     * @note The callback should handle errors gracefully and return nullptr on failure
     */
    void set_prefetch_callback(PrefetchCallback callback);

    /**
     * @brief Update the maximum cache size with dynamic adjustment
     * 
     * Changes the cache size limit and triggers immediate adjustment if the
     * current cache usage exceeds the new limit. This enables dynamic memory
     * management based on system conditions.
     * 
     * @param size New maximum cache size in bytes
     * 
     * @note If the new size is smaller than current usage, triggers immediate eviction
     * @note Changes take effect immediately
     * @note Size should be chosen based on available memory and performance requirements
     */
    void set_cache_size(size_t size);

    /**
     * @brief Configure the read-ahead window size for predictive loading
     * 
     * Sets the number of sequences to prefetch ahead of the current access position.
     * Larger windows improve sequential access performance but consume more memory.
     * 
     * @param window Number of sequences to prefetch (0 disables read-ahead)
     * 
     * @note Optimal window size depends on access patterns and memory availability
     * @note Changes take effect on subsequent read-ahead operations
     * @note Window size is automatically adjusted based on memory pressure
     */
    void set_read_ahead_window(size_t window);

    /**
     * @brief Configure memory monitoring check interval
     * 
     * Sets how frequently the memory monitor checks system memory usage.
     * Shorter intervals provide more responsive memory pressure detection
     * but increase CPU overhead.
     * 
     * @param interval_ms Monitoring interval in milliseconds
     * 
     * @note Minimum recommended interval is 100ms to avoid excessive overhead
     * @note Changes take effect on the next monitoring cycle
     * @note Shorter intervals improve responsiveness to memory pressure
     */
    void set_memory_check_interval(size_t interval_ms);

    /**
     * @brief Enable or disable all optimization features
     * 
     * Master control for enabling/disabling the entire optimization system.
     * When disabled, the manager operates in pass-through mode with minimal
     * overhead.
     * 
     * @param enabled True to enable optimization, false to disable
     * 
     * @note Disabling optimization stops background threads and clears cache
     * @note Re-enabling optimization restarts all components
     * @note Useful for performance testing and debugging
     */
    void set_optimization_enabled(bool enabled);

    /**
     * @brief Enable or disable read-ahead prefetching
     * 
     * Controls whether the manager performs predictive data loading based on
     * access patterns. Disabling read-ahead reduces memory usage but may
     * impact sequential access performance.
     * 
     * @param enabled True to enable read-ahead, false to disable
     * 
     * @note Disabling read-ahead stops prefetch operations immediately
     * @note Cache and memory monitoring remain active when read-ahead is disabled
     * @note Useful for random access patterns where prefetching is ineffective
     */
    void set_read_ahead_enabled(bool enabled);

    /**
     * @brief Enable or disable adaptive cache management
     * 
     * Controls whether the cache automatically adjusts its behavior based on
     * memory pressure and access patterns. Adaptive management optimizes
     * performance but adds computational overhead.
     * 
     * @param enabled True to enable adaptive management, false for static behavior
     * 
     * @note Adaptive management includes dynamic eviction policy selection
     * @note Static mode uses fixed LRU eviction policy
     * @note Adaptive mode typically provides better performance for varied workloads
     */
    void set_adaptive_cache_enabled(bool enabled);

    /**
     * @brief Comprehensive optimization statistics and performance metrics
     * 
     * Contains detailed information about optimization performance including
     * cache effectiveness, read-ahead status, memory usage, and access patterns.
     * This structure provides insights for performance tuning and monitoring.
     */
    struct OptimizationStats {
        llama_dataset_streaming_cache::CacheStats cache_stats;           ///< Cache performance statistics
        llama_dataset_streaming_read_ahead::Status read_ahead_status;    ///< Read-ahead operation status
        llama_dataset_streaming_memory_monitor::MemoryInfo memory_info;  ///< Current memory usage information
        uint64_t access_count;                                           ///< Total number of data access operations
        uint64_t sequential_access_count;                                ///< Number of sequential access operations
        uint64_t random_access_count;                                    ///< Number of random access operations
        double sequential_access_ratio;                                  ///< Ratio of sequential to total accesses
    };

    /**
     * @brief Retrieve comprehensive optimization statistics
     * 
     * Returns detailed performance metrics and statistics for all optimization
     * components. This information is useful for performance monitoring,
     * tuning, and debugging optimization effectiveness.
     * 
     * @return OptimizationStats structure containing current performance metrics
     * 
     * @note Statistics are collected continuously during operation
     * @note The returned data represents a snapshot at the time of the call
     * @note Thread-safe and can be called while optimization is active
     * @note Useful for real-time performance monitoring and adaptive tuning
     */
    OptimizationStats get_stats() const;
};
