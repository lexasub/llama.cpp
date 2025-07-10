#pragma once

#include "streaming-cache.h"
#include "streaming-read-ahead.h"
#include "streaming-memory-monitor.h"

#include <memory>
#include <string>
#include <functional>

/**
 * @brief Streaming optimization manager
 * 
 * This class integrates all streaming optimization components:
 * - Cache management
 * - Read-ahead buffering
 * - Memory monitoring
 * - Performance tracking
 */
class StreamingOptimizationManager {
public:
    using PrefetchCallback = std::function<void*(uint64_t, size_t*)>;
    
private:
    std::unique_ptr<StreamingCache> cache;
    std::unique_ptr<StreamingReadAhead> read_ahead;
    std::unique_ptr<StreamingMemoryMonitor> memory_monitor;
    
    PrefetchCallback prefetch_callback;
    std::string dataset_name;
    bool optimization_enabled;
    
    // Performance tracking
    uint64_t access_count;
    uint64_t sequential_access_count;
    uint64_t random_access_count;
    uint64_t last_accessed_id;
    
    void handle_memory_pressure(double pressure);
    void handle_prefetch(uint64_t sequence_id);
    void update_access_pattern(uint64_t sequence_id);
    
public:
    StreamingOptimizationManager(const std::string& name = "unnamed");
    ~StreamingOptimizationManager();
    
    // Initialize the optimization manager
    void initialize(size_t cache_size = 64 * 1024 * 1024);
    
    // Start optimization
    void start();
    
    // Stop optimization
    void stop();
    
    // Get cached data for a sequence
    void* get_sequence(uint64_t sequence_id);
    
    // Put data into cache
    void put_sequence(uint64_t sequence_id, void* data, size_t size);
    
    // Set the prefetch callback
    void set_prefetch_callback(PrefetchCallback callback);
    
    // Configure optimization components
    void set_cache_size(size_t size);
    void set_read_ahead_window(size_t window);
    void set_memory_check_interval(size_t interval_ms);
    
    // Enable/disable optimization features
    void set_optimization_enabled(bool enabled);
    void set_read_ahead_enabled(bool enabled);
    void set_adaptive_cache_enabled(bool enabled);
    
    // Get optimization statistics
    struct OptimizationStats {
        StreamingCache::CacheStats cache_stats;
        StreamingReadAhead::Status read_ahead_status;
        StreamingMemoryMonitor::MemoryInfo memory_info;
        uint64_t access_count;
        uint64_t sequential_access_count;
        uint64_t random_access_count;
        double sequential_access_ratio;
    };
    
    OptimizationStats get_stats() const;
};