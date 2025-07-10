#pragma once

#include <cstddef>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

/**
 * @brief Memory usage monitor for streaming datasets
 * 
 * This class monitors system memory usage and provides callbacks
 * when memory pressure is detected.
 */
class StreamingMemoryMonitor {
public:
    using MemoryPressureCallback = std::function<void(double pressure_level)>;
    
private:
    std::thread monitor_thread;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> running;
    
    size_t check_interval_ms;
    double high_pressure_threshold;
    double low_pressure_threshold;
    MemoryPressureCallback pressure_callback;
    
    void monitor_loop();
    double get_memory_pressure();
    
public:
    StreamingMemoryMonitor(
        size_t interval_ms = 1000,
        double high_threshold = 0.8,
        double low_threshold = 0.6);
    ~StreamingMemoryMonitor();
    
    // Start monitoring
    void start();
    
    // Stop monitoring
    void stop();
    
    // Set the memory pressure callback
    void set_pressure_callback(MemoryPressureCallback callback);
    
    // Set monitoring parameters
    void set_check_interval(size_t interval_ms);
    void set_pressure_thresholds(double high_threshold, double low_threshold);
    
    // Get current memory usage information
    struct MemoryInfo {
        size_t total_physical_memory;
        size_t available_physical_memory;
        size_t process_physical_memory;
        double memory_pressure;
    };
    
    MemoryInfo get_memory_info() const;
};