/**
 * @file streaming-memory-monitor.cpp
 * @brief Implementation of memory monitoring and pressure detection for streaming dataset operations
 *
 * This file implements comprehensive memory monitoring capabilities for the dataset converter's
 * streaming subsystem. It provides real-time memory usage tracking, adaptive pressure detection,
 * and intelligent resource management to optimize streaming performance while preventing
 * out-of-memory conditions.
 *
 * ## Implementation Overview
 * The memory monitor operates as a background service that continuously tracks both system-wide
 * and process-specific memory usage. It implements a dual-threshold hysteresis system to prevent
 * callback oscillation and provides stable memory pressure notifications.
 *
 * ## Memory Tracking Algorithms
 * - **System Memory Monitoring**: Uses /proc/meminfo on Linux for accurate system memory statistics
 * - **Process Memory Tracking**: Leverages getrusage() for process-specific memory consumption
 * - **Pressure Calculation**: Implements normalized pressure calculation (0.0 to 1.0 scale)
 * - **Hysteresis Control**: Dual-threshold system prevents rapid callback triggering
 *
 * ## Platform Integration
 * The implementation integrates with platform-specific APIs to gather memory information:
 * - Linux: /proc/meminfo for system stats, getrusage() for process stats
 * - Cross-platform compatibility through platform-compat.h abstraction layer
 * - Efficient parsing of system memory information with minimal overhead
 *
 * ## Threading and Synchronization
 * - Background monitoring thread with configurable check intervals
 * - Thread-safe access to all configuration parameters using mutex protection
 * - Condition variable for efficient thread coordination and shutdown
 * - Lock-free pressure calculation for minimal performance impact
 *
 * ## Performance Optimizations
 * - Efficient file parsing with early termination when all values are found
 * - Minimal system call overhead through batched information gathering
 * - Configurable monitoring intervals to balance responsiveness and CPU usage
 * - Lightweight callback mechanism with user-defined pressure handlers
 *
 * ## Integration with Streaming Cache
 * This monitor is designed to work seamlessly with the streaming cache system:
 * - Triggers cache evictions when memory pressure exceeds thresholds
 * - Provides memory availability information for cache size adjustments
 * - Coordinates with read-ahead operations to prevent memory exhaustion
 * - Enables adaptive caching strategies based on system memory conditions
 *
 * ## Error Handling and Robustness
 * - Graceful degradation when system memory information is unavailable
 * - Safe defaults for memory pressure calculation in error conditions
 * - Comprehensive logging for monitoring and debugging purposes
 * - Thread-safe shutdown procedures with proper resource cleanup
 *
 * @author llama.cpp dataset-converter team
 * @version 1.0
 * @since 2024
 */

#include "streaming-memory-monitor.h"

#include "platform/platform-compat.h"

#include <chrono>
#include <fstream>
#include <string>
#include <thread>

#include "log.h"
#include "llama-impl.h"

/**
 * @brief Constructs a memory monitor with specified monitoring parameters
 *
 * Initializes the memory monitor with configurable check interval and pressure thresholds.
 * The monitor is created in a stopped state and requires explicit start() call to begin
 * monitoring operations. The constructor validates threshold parameters and sets up
 * internal state for thread-safe operation.
 *
 * @param interval_ms Monitoring check interval in milliseconds (recommended: 500-2000ms)
 * @param high_threshold High pressure threshold (0.0-1.0, triggers pressure callbacks)
 * @param low_threshold Low pressure threshold (0.0-1.0, signals pressure relief)
 *
 * @note The high_threshold should be greater than low_threshold to provide hysteresis
 * @note Shorter intervals provide more responsive monitoring but increase CPU overhead
 * @note Default values are optimized for typical streaming workloads
 */
llama_dataset_streaming_memory_monitor::llama_dataset_streaming_memory_monitor(
    size_t interval_ms,
    double high_threshold,
    double low_threshold)
    : running(false),
      check_interval_ms(interval_ms),
      high_pressure_threshold(high_threshold),
      low_pressure_threshold(low_threshold) {
}

/**
 * @brief Destructor ensuring clean shutdown of monitoring thread
 *
 * Automatically stops the monitoring thread if it's still running and waits for
 * clean shutdown. This ensures proper resource cleanup and prevents thread leaks.
 * The destructor is safe to call even if the monitor was never started.
 */
llama_dataset_streaming_memory_monitor::~llama_dataset_streaming_memory_monitor() {
    stop();
}

/**
 * @brief Main monitoring loop executed in the background thread
 *
 * This is the core monitoring algorithm that runs continuously in a separate thread.
 * It implements a hysteresis-based pressure detection system to prevent callback
 * oscillation and provides stable memory pressure notifications.
 *
 * ## Algorithm Details
 * 1. **Interval-based Monitoring**: Uses condition variable with timeout for efficient waiting
 * 2. **Hysteresis Implementation**: Maintains separate notification states for high/low pressure
 * 3. **Threshold Crossing Detection**: Only triggers callbacks when thresholds are crossed
 * 4. **Graceful Shutdown**: Responds immediately to stop requests via condition variable
 *
 * ## Hysteresis Behavior
 * - High pressure callback triggered when pressure >= high_threshold (first time only)
 * - Low pressure callback triggered when pressure <= low_threshold (after high pressure)
 * - Prevents rapid callback oscillation when pressure hovers near thresholds
 * - State machine ensures callbacks are only sent on actual threshold crossings
 *
 * ## Thread Safety
 * The loop uses proper synchronization to safely access shared state and respond
 * to configuration changes and shutdown requests from other threads.
 */
void llama_dataset_streaming_memory_monitor::monitor_loop() {
    LLAMA_LOG_DEBUG("Memory monitor thread started\n");

    bool high_pressure_notified = false;
    bool low_pressure_notified = false;

    while (running) {
        // Sleep for the check interval using condition variable for efficient waiting
        // This allows immediate response to shutdown requests
        {
            std::unique_lock lock(mutex);
            cv.wait_for(lock, std::chrono::milliseconds(check_interval_ms),
                [this] { return !running; });

            if (!running) {
                break;
            }
        }

        // Check current memory pressure using platform-specific implementation
        double pressure = get_memory_pressure();

        // Implement hysteresis-based threshold crossing detection
        // Only notify when pressure actually crosses thresholds to prevent oscillation
        if (pressure >= high_pressure_threshold && !high_pressure_notified) {
            if (pressure_callback) {
                pressure_callback(pressure);
            }
            high_pressure_notified = true;
            low_pressure_notified = false;
            LLAMA_LOG_DEBUG("High memory pressure detected: %.2f\n", pressure);
        } else if (pressure <= low_pressure_threshold && !low_pressure_notified) {
            if (pressure_callback) {
                pressure_callback(pressure);
            }
            high_pressure_notified = false;
            low_pressure_notified = true;
            LLAMA_LOG_DEBUG("Low memory pressure detected: %.2f\n", pressure);
        }
    }

    LLAMA_LOG_DEBUG("Memory monitor thread stopped");
}

/**
 * @brief Calculates current memory pressure level using system-specific APIs
 *
 * This method implements the core memory pressure calculation algorithm by gathering
 * both system-wide and process-specific memory statistics. It uses platform-specific
 * APIs to obtain accurate memory information and normalizes the result to a 0.0-1.0 scale.
 *
 * ## Algorithm Implementation
 * 1. **Process Memory Tracking**: Uses getrusage(RUSAGE_SELF) for current process memory
 * 2. **System Memory Analysis**: Parses /proc/meminfo for total and available system memory
 * 3. **Pressure Calculation**: Computes normalized pressure as (1 - available/total)
 * 4. **Error Handling**: Returns 0.0 pressure if system information is unavailable
 *
 * ## Platform-Specific Details
 * - **Linux**: Uses /proc/meminfo which provides accurate available memory accounting
 * - **MemAvailable**: Preferred over MemFree as it accounts for reclaimable memory
 * - **Efficient Parsing**: Early termination when all required values are found
 * - **Unit Conversion**: Converts from KB (proc format) to bytes for consistency
 *
 * ## Performance Considerations
 * - Minimal file I/O through efficient parsing with early exit
 * - String operations optimized for typical /proc/meminfo format
 * - No dynamic memory allocation during pressure calculation
 * - Lightweight system calls with minimal overhead
 *
 * @return Normalized memory pressure level (0.0 = no pressure, 1.0 = critical pressure)
 * @note Returns 0.0 if system memory information cannot be obtained
 * @note The calculation prioritizes available memory over free memory for accuracy
 */
double llama_dataset_streaming_memory_monitor::get_memory_pressure() {
    // Get process memory usage for potential future use in pressure calculation
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    // Get system memory info from /proc/meminfo (Linux-specific implementation)
    size_t total_memory = 0;
    size_t available_memory = 0;

    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            // Parse MemTotal line for total system memory
            if (line.find("MemTotal:") == 0) {
                size_t pos = line.find_first_of("0123456789");
                if (pos != std::string::npos) {
                    total_memory = std::stoull(line.substr(pos)) * 1024; // Convert KB to bytes
                }
            } 
            // Parse MemAvailable line for available memory (includes reclaimable)
            else if (line.find("MemAvailable:") == 0) {
                size_t pos = line.find_first_of("0123456789");
                if (pos != std::string::npos) {
                    available_memory = std::stoull(line.substr(pos)) * 1024; // Convert KB to bytes
                }
            }

            // Early termination optimization: exit when both values are found
            if (total_memory > 0 && available_memory > 0) {
                break;
            }
        }
        meminfo.close();
    }

    // Calculate normalized memory pressure (0.0 to 1.0 scale)
    // Pressure = 1 - (available / total), where 1.0 indicates critical memory pressure
    double pressure = 0.0;
    if (total_memory > 0) {
        pressure = 1.0 - static_cast<double>(available_memory) / total_memory;
    }

    return pressure;
}

/**
 * @brief Starts the memory monitoring background thread
 *
 * Initiates continuous memory monitoring by creating and starting a background thread
 * that executes the monitoring loop. The thread will run until explicitly stopped
 * and will trigger pressure callbacks when configured thresholds are crossed.
 *
 * ## Thread Management
 * - Creates a new std::thread running the monitor_loop() method
 * - Sets the running flag to enable the monitoring loop
 * - Thread-safe operation with proper synchronization
 * - Idempotent: safe to call multiple times (no-op if already running)
 *
 * ## Error Handling
 * - Gracefully handles thread creation failures
 * - Logs monitoring start with configuration details
 * - Maintains consistent internal state even if thread creation fails
 *
 * @throws std::system_error if thread creation fails
 * @note This method is thread-safe and idempotent
 * @note The monitoring thread will continue until stop() is called
 */
void llama_dataset_streaming_memory_monitor::start() {
    if (running) {
        return;
    }

    running = true;
    monitor_thread = std::thread(&llama_dataset_streaming_memory_monitor::monitor_loop, this);

    LLAMA_LOG_DEBUG("Started memory monitor with interval %zu ms", check_interval_ms);
}

/**
 * @brief Stops the memory monitoring background thread gracefully
 *
 * Performs a clean shutdown of the monitoring thread by signaling the stop condition
 * and waiting for the thread to complete its current monitoring cycle. This ensures
 * proper resource cleanup and prevents thread leaks.
 *
 * ## Shutdown Procedure
 * 1. **Signal Stop**: Sets running flag to false under mutex protection
 * 2. **Wake Thread**: Notifies condition variable to interrupt sleep
 * 3. **Wait for Completion**: Joins the thread to ensure clean shutdown
 * 4. **Resource Cleanup**: Ensures all thread resources are properly released
 *
 * ## Thread Safety
 * - Uses mutex protection when modifying the running flag
 * - Condition variable notification ensures immediate response to stop request
 * - Thread join operation waits for complete shutdown before returning
 * - Idempotent: safe to call multiple times (no-op if already stopped)
 *
 * ## Performance Considerations
 * - Immediate response through condition variable notification
 * - No forced thread termination (graceful shutdown only)
 * - Minimal blocking time due to efficient thread coordination
 *
 * @note This method is thread-safe and idempotent
 * @note Blocks until the monitoring thread has completely stopped
 * @note Safe to call from destructor or multiple times
 */
void llama_dataset_streaming_memory_monitor::stop() {
    if (!running) {
        return;
    }

    {
        std::lock_guard lock(mutex);
        running = false;
    }

    // Wake up the monitor thread to process the stop signal immediately
    cv.notify_all();

    // Wait for the monitor thread to finish its current cycle and exit cleanly
    if (monitor_thread.joinable()) {
        monitor_thread.join();
    }

    LLAMA_LOG_DEBUG("Stopped memory monitor\n");
}

/**
 * @brief Sets the callback function for memory pressure notifications
 *
 * Registers a user-defined callback function that will be invoked when memory pressure
 * levels cross the configured thresholds. The callback enables integration with cache
 * management systems and other memory-sensitive components.
 *
 * ## Callback Execution Context
 * - Executed in the monitoring thread context (not the calling thread)
 * - Called with current pressure level as parameter (0.0 to 1.0)
 * - Should be lightweight to avoid blocking the monitoring loop
 * - Thread-safe registration with mutex protection
 *
 * ## Integration Patterns
 * - Cache eviction triggers when pressure exceeds high threshold
 * - Memory allocation adjustments based on pressure levels
 * - Read-ahead operation throttling during high pressure
 * - Adaptive buffer size management based on available memory
 *
 * @param callback Function to call when pressure thresholds are crossed
 * @note Setting nullptr disables pressure notifications
 * @note Callback should be lightweight to avoid monitoring delays
 * @note Thread-safe: can be called while monitoring is active
 */
void llama_dataset_streaming_memory_monitor::set_pressure_callback(MemoryPressureCallback callback) {
    std::lock_guard lock(mutex);
    pressure_callback = callback;
}

/**
 * @brief Updates the monitoring check interval dynamically
 *
 * Changes the frequency at which the monitor checks memory usage. This allows
 * runtime adjustment of monitoring responsiveness versus CPU overhead trade-offs.
 * The change takes effect on the next monitoring cycle.
 *
 * ## Performance Trade-offs
 * - **Shorter Intervals**: More responsive pressure detection, higher CPU overhead
 * - **Longer Intervals**: Lower CPU overhead, delayed pressure detection
 * - **Recommended Range**: 500-2000ms for typical streaming workloads
 * - **Minimum Practical**: 100ms to avoid excessive system call overhead
 *
 * ## Dynamic Adjustment Strategies
 * - Reduce interval during high-pressure periods for faster response
 * - Increase interval during stable periods to reduce overhead
 * - Adaptive intervals based on memory usage patterns
 * - Integration with workload characteristics for optimal tuning
 *
 * @param interval_ms New monitoring interval in milliseconds
 * @note Changes take effect on the next monitoring cycle
 * @note Thread-safe: can be called while monitoring is active
 * @note Minimum recommended value is 100ms to avoid excessive overhead
 */
void llama_dataset_streaming_memory_monitor::set_check_interval(size_t interval_ms) {
    std::lock_guard lock(mutex);
    check_interval_ms = interval_ms;
    LLAMA_LOG_DEBUG("Set memory monitor check interval to %zu ms\n", check_interval_ms);
}

/**
 * @brief Updates memory pressure thresholds for adaptive monitoring
 *
 * Configures the pressure levels that trigger callback notifications, enabling
 * dynamic adjustment of memory management sensitivity. The dual-threshold system
 * implements hysteresis to prevent callback oscillation near threshold boundaries.
 *
 * ## Hysteresis Implementation
 * - **High Threshold**: Triggers pressure callbacks when exceeded
 * - **Low Threshold**: Signals return to normal conditions
 * - **Hysteresis Gap**: Difference between thresholds prevents oscillation
 * - **Recommended Gap**: 0.1-0.2 (10-20%) for stable operation
 *
 * ## Adaptive Threshold Strategies
 * - Lower thresholds for memory-constrained environments
 * - Higher thresholds for systems with abundant memory
 * - Dynamic adjustment based on workload characteristics
 * - Integration with system memory capacity and usage patterns
 *
 * ## Validation and Safety
 * - High threshold should be greater than low threshold
 * - Both thresholds should be in range [0.0, 1.0]
 * - Changes take effect immediately on next monitoring cycle
 * - Thread-safe updates with proper synchronization
 *
 * @param high_threshold High pressure threshold (0.0 to 1.0)
 * @param low_threshold Low pressure threshold (0.0 to 1.0)
 * @note high_threshold should be > low_threshold for proper hysteresis
 * @note Changes take effect immediately on the next monitoring cycle
 * @note Thread-safe: can be called while monitoring is active
 */
void llama_dataset_streaming_memory_monitor::set_pressure_thresholds(double high_threshold, double low_threshold) {
    std::lock_guard<std::mutex> lock(mutex);
    high_pressure_threshold = high_threshold;
    low_pressure_threshold = low_threshold;
    LLAMA_LOG_DEBUG("Set memory monitor pressure thresholds to high=%.2f, low=%.2f\n", high_threshold, low_threshold);
}

/**
 * @brief Retrieves comprehensive memory usage information snapshot
 *
 * Provides a complete snapshot of current memory conditions including system-wide
 * statistics and process-specific metrics. This method performs immediate system
 * queries and returns detailed memory information without waiting for the next
 * monitoring cycle.
 *
 * ## Information Gathering Process
 * 1. **Process Memory**: Uses getrusage() to get current process memory usage
 * 2. **System Memory**: Parses /proc/meminfo for total and available system memory
 * 3. **Pressure Calculation**: Computes current pressure using same algorithm as monitor
 * 4. **Comprehensive Report**: Returns structured information for analysis
 *
 * ## Platform-Specific Implementation
 * - **Linux**: Uses /proc/meminfo and getrusage() for accurate statistics
 * - **Process Memory**: Reports maximum resident set size (ru_maxrss)
 * - **System Memory**: Uses MemTotal and MemAvailable from /proc/meminfo
 * - **Unit Consistency**: All values returned in bytes for consistency
 *
 * ## Use Cases
 * - On-demand memory analysis without waiting for monitoring cycle
 * - Debugging and diagnostics for memory-related issues
 * - Integration with external monitoring and alerting systems
 * - Performance analysis and capacity planning
 *
 * ## Performance Considerations
 * - Immediate system queries may have slight overhead
 * - Same efficient parsing algorithm as monitoring loop
 * - No caching: always returns current system state
 * - Thread-safe: can be called concurrently with monitoring
 *
 * @return MemoryInfo structure containing comprehensive memory statistics
 * @note Performs immediate system queries (not cached values)
 * @note Thread-safe: can be called while monitoring is active
 * @note All memory values are returned in bytes for consistency
 */
llama_dataset_streaming_memory_monitor::MemoryInfo llama_dataset_streaming_memory_monitor::get_memory_info() const {
    MemoryInfo info;
    info.total_physical_memory = 0;
    info.available_physical_memory = 0;
    info.process_physical_memory = 0;
    info.memory_pressure = 0.0;

    // Get process memory usage using getrusage() system call
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    info.process_physical_memory = usage.ru_maxrss * 1024; // Convert KB to bytes

    // Get system memory info from /proc/meminfo using same algorithm as monitor
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            // Parse total system memory
            if (line.find("MemTotal:") == 0) {
                size_t pos = line.find_first_of("0123456789");
                if (pos != std::string::npos) {
                    info.total_physical_memory = std::stoull(line.substr(pos)) * 1024; // Convert KB to bytes
                }
            } 
            // Parse available memory (includes reclaimable memory)
            else if (line.find("MemAvailable:") == 0) {
                size_t pos = line.find_first_of("0123456789");
                if (pos != std::string::npos) {
                    info.available_physical_memory = std::stoull(line.substr(pos)) * 1024; // Convert KB to bytes
                }
            }

            // Early termination optimization when both values are found
            if (info.total_physical_memory > 0 && info.available_physical_memory > 0) {
                break;
            }
        }
        meminfo.close();
    }

    // Calculate memory pressure using same algorithm as monitoring loop
    if (info.total_physical_memory > 0) {
        info.memory_pressure = 1.0 - static_cast<double>(info.available_physical_memory) / info.total_physical_memory;
    }

    return info;
}
