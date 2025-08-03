#pragma once

#include <cstddef>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

/**
 * @file streaming-memory-monitor.h
 * @brief Memory monitoring and pressure detection for streaming dataset operations
 *
 * This module provides comprehensive memory monitoring capabilities for the dataset converter's
 * streaming subsystem. It implements real-time memory usage tracking, adaptive pressure detection,
 * and callback-based notification system to enable intelligent cache management and resource
 * optimization.
 *
 * ## Key Features
 * - **Real-time Memory Monitoring**: Continuous tracking of system and process memory usage
 * - **Adaptive Pressure Detection**: Configurable thresholds for memory pressure levels
 * - **Callback-based Notifications**: Event-driven architecture for memory pressure responses
 * - **Cross-platform Compatibility**: Platform-specific memory information gathering
 * - **Thread-safe Operations**: Safe concurrent access to monitoring data and controls
 *
 * ## Memory Pressure Management
 * The monitor operates with dual thresholds:
 * - High pressure threshold: Triggers aggressive cache eviction and resource cleanup
 * - Low pressure threshold: Signals return to normal memory conditions
 *
 * ## Integration with Streaming Cache
 * This monitor works closely with the streaming cache system to:
 * - Trigger cache evictions when memory pressure is detected
 * - Adjust cache size limits based on available memory
 * - Coordinate read-ahead operations with memory availability
 * - Optimize memory allocation patterns for streaming workloads
 *
 * ## Performance Considerations
 * - Configurable monitoring intervals to balance responsiveness and overhead
 * - Efficient platform-specific memory queries
 * - Minimal impact on streaming performance through background monitoring
 * - Adaptive algorithms that learn from memory usage patterns
 *
 * @author llama.cpp dataset-converter team
 * @version 1.0
 * @since 2024
 */

/**
 * @brief Memory usage monitor for streaming datasets with adaptive pressure detection
 *
 * This class provides comprehensive memory monitoring capabilities for streaming dataset
 * operations. It continuously tracks system memory usage, detects memory pressure conditions,
 * and triggers appropriate responses through a callback mechanism. The monitor is designed
 * to work seamlessly with the streaming cache system to optimize memory usage and prevent
 * out-of-memory conditions during large dataset processing.
 *
 * The monitor implements a dual-threshold system where different pressure levels trigger
 * different response strategies, enabling fine-grained control over memory management.
 *
 * ## Thread Safety
 * All public methods are thread-safe and can be called concurrently from multiple threads.
 * The internal monitoring loop runs in a separate background thread.
 *
 * ## Usage Example
 * ```cpp
 * auto monitor = std::make_unique<llama_dataset_streaming_memory_monitor>(500, 0.85, 0.65);
 * monitor->set_pressure_callback([&cache](double pressure) {
 *     if (pressure > 0.8) {
 *         cache.evict_lru();
 *     }
 * });
 * monitor->start();
 * ```
 */
class llama_dataset_streaming_memory_monitor {
public:
    /**
     * @brief Callback function type for memory pressure notifications
     * 
     * This callback is invoked when memory pressure levels change. The pressure_level
     * parameter ranges from 0.0 (no pressure) to 1.0 (maximum pressure).
     * 
     * @param pressure_level Current memory pressure level (0.0 to 1.0)
     */
    using MemoryPressureCallback = std::function<void(double pressure_level)>;

private:
    std::thread monitor_thread;           ///< Background monitoring thread
    std::mutex mutex;                     ///< Synchronization mutex for thread-safe operations
    std::condition_variable cv;           ///< Condition variable for thread coordination
    volatile bool running;                ///< Flag indicating if monitoring is active

    size_t check_interval_ms;             ///< Monitoring interval in milliseconds
    double high_pressure_threshold;       ///< High pressure threshold (0.0 to 1.0)
    double low_pressure_threshold;        ///< Low pressure threshold (0.0 to 1.0)
    MemoryPressureCallback pressure_callback; ///< User-defined pressure callback function

    /**
     * @brief Main monitoring loop executed in background thread
     * 
     * This method continuously monitors memory usage at the specified interval,
     * calculates pressure levels, and triggers callbacks when thresholds are crossed.
     * The loop implements hysteresis to prevent callback oscillation.
     */
    void monitor_loop();

    /**
     * @brief Calculate current memory pressure level
     * 
     * Analyzes system memory usage and returns a normalized pressure value.
     * The calculation considers both system-wide and process-specific memory usage.
     * 
     * @return Memory pressure level from 0.0 (no pressure) to 1.0 (critical pressure)
     */
    double get_memory_pressure();

public:
    /**
     * @brief Construct a new memory monitor with specified parameters
     * 
     * Creates a memory monitor with configurable monitoring interval and pressure
     * thresholds. The monitor is created in a stopped state and must be explicitly
     * started using the start() method.
     * 
     * @param interval_ms Monitoring check interval in milliseconds (default: 1000ms)
     * @param high_threshold High pressure threshold 0.0-1.0 (default: 0.8)
     * @param low_threshold Low pressure threshold 0.0-1.0 (default: 0.6)
     * 
     * @note high_threshold should be greater than low_threshold to provide hysteresis
     * @note Shorter intervals provide more responsive monitoring but increase CPU overhead
     */
    llama_dataset_streaming_memory_monitor(
        size_t interval_ms = 1000,
        double high_threshold = 0.8,
        double low_threshold = 0.6);

    /**
     * @brief Destructor - ensures monitoring thread is properly stopped
     * 
     * Automatically stops the monitoring thread if it's still running and waits
     * for clean shutdown before destroying the object.
     */
    ~llama_dataset_streaming_memory_monitor();

    /**
     * @brief Start the memory monitoring background thread
     * 
     * Begins continuous memory monitoring in a separate thread. The monitor will
     * check memory usage at the configured interval and trigger callbacks when
     * pressure thresholds are crossed.
     * 
     * @throws std::runtime_error if monitoring is already running
     * @throws std::system_error if thread creation fails
     * 
     * @note This method is thread-safe and can be called from any thread
     */
    void start();

    /**
     * @brief Stop the memory monitoring background thread
     * 
     * Gracefully stops the monitoring thread and waits for it to complete.
     * After this call, no further pressure callbacks will be triggered until
     * start() is called again.
     * 
     * @note This method is thread-safe and can be called from any thread
     * @note Calling stop() on an already stopped monitor is safe (no-op)
     */
    void stop();

    /**
     * @brief Set the callback function for memory pressure notifications
     * 
     * Registers a callback function that will be invoked when memory pressure
     * levels cross the configured thresholds. The callback receives the current
     * pressure level as a parameter.
     * 
     * @param callback Function to call when pressure thresholds are crossed
     * 
     * @note The callback is executed in the monitoring thread context
     * @note Callback should be lightweight to avoid blocking the monitor
     * @note Setting a null callback disables pressure notifications
     */
    void set_pressure_callback(MemoryPressureCallback callback);

    /**
     * @brief Update the monitoring check interval
     * 
     * Changes how frequently the monitor checks memory usage. Shorter intervals
     * provide more responsive monitoring but increase CPU overhead.
     * 
     * @param interval_ms New monitoring interval in milliseconds
     * 
     * @note Changes take effect on the next monitoring cycle
     * @note Minimum recommended interval is 100ms to avoid excessive overhead
     */
    void set_check_interval(size_t interval_ms);

    /**
     * @brief Update memory pressure thresholds
     * 
     * Configures the pressure levels that trigger callback notifications.
     * The high threshold triggers pressure callbacks, while the low threshold
     * signals return to normal conditions (hysteresis).
     * 
     * @param high_threshold High pressure threshold (0.0 to 1.0)
     * @param low_threshold Low pressure threshold (0.0 to 1.0)
     * 
     * @pre high_threshold > low_threshold (for proper hysteresis)
     * @pre Both thresholds must be in range [0.0, 1.0]
     * 
     * @note Changes take effect immediately on the next monitoring cycle
     */
    void set_pressure_thresholds(double high_threshold, double low_threshold);

    /**
     * @brief Comprehensive memory usage information structure
     * 
     * Contains detailed information about current memory usage including
     * system-wide statistics and process-specific metrics.
     */
    struct MemoryInfo {
        size_t total_physical_memory;      ///< Total system physical memory in bytes
        size_t available_physical_memory;  ///< Available physical memory in bytes
        size_t process_physical_memory;    ///< Current process physical memory usage in bytes
        double memory_pressure;            ///< Current memory pressure level (0.0 to 1.0)
    };

    /**
     * @brief Get current memory usage information
     * 
     * Retrieves comprehensive memory usage statistics including system-wide
     * and process-specific metrics. This method provides a snapshot of current
     * memory conditions without waiting for the next monitoring cycle.
     * 
     * @return MemoryInfo structure containing current memory statistics
     * 
     * @note This method performs immediate system queries and may have slight overhead
     * @note The returned pressure level uses the same calculation as the monitoring loop
     * @note This method is thread-safe and can be called while monitoring is active
     */
    MemoryInfo get_memory_info() const;
};
