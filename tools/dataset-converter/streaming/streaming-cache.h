#pragma once

#include <unordered_map>
#include <list>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <mutex>
#include <atomic>
#include <memory>
#include <algorithm>

/**
 * @brief Enhanced LRU Cache for streaming dataset sequences
 * 
 * This cache manages memory usage for streaming datasets by keeping
 * frequently accessed sequences in memory while evicting older ones.
 * Features include:
 * - Thread-safe operations
 * - Adaptive memory management
 * - Read-ahead buffering for sequential access
 * - Memory pressure detection
 * - Performance statistics tracking
 */
class StreamingCache {
public:
    struct CacheEntry {
        void* data;
        size_t size;
        uint64_t sequence_id;
        uint64_t last_access_time;
        uint32_t access_count;
        
        CacheEntry(void* d, size_t s, uint64_t id) 
            : data(d), size(s), sequence_id(id), last_access_time(0), access_count(0) {}
    };
    
    enum class EvictionPolicy {
        LRU,        // Least Recently Used
        LFU,        // Least Frequently Used
        ADAPTIVE    // Combination based on access patterns
    };
    
private:
    size_t max_memory_bytes;
    std::atomic<size_t> current_memory_bytes;
    std::list<CacheEntry> cache_list;
    std::unordered_map<uint64_t, std::list<CacheEntry>::iterator> cache_map;
    std::mutex cache_mutex;
    
    // Read-ahead buffering
    bool read_ahead_enabled;
    size_t read_ahead_window;
    std::vector<uint64_t> prefetch_queue;
    
    // Performance tracking
    uint64_t hit_count;
    uint64_t miss_count;
    uint64_t eviction_count;
    uint64_t access_count;
    uint64_t timestamp_counter;
    
    // Adaptive memory management
    EvictionPolicy eviction_policy;
    double memory_pressure_threshold;
    bool adaptive_sizing_enabled;
    size_t initial_max_memory;
    
    void evict_lru();
    void evict_lfu();
    void evict_adaptive();
    void update_access_stats(std::list<CacheEntry>::iterator entry_it);
    void check_memory_pressure();
    void prefetch_sequences(uint64_t current_id);
    uint64_t get_current_timestamp();
    
public:
    explicit StreamingCache(size_t max_memory = 64 * 1024 * 1024); // Default 64MB
    ~StreamingCache();
    
    // Get cached data for a sequence (returns nullptr if not cached)
    void* get(uint64_t sequence_id);
    
    // Put data into cache (takes ownership of data pointer)
    void put(uint64_t sequence_id, void* data, size_t size);
    
    // Remove specific sequence from cache
    void remove(uint64_t sequence_id);
    
    // Clear all cached data
    void clear();
    
    // Get cache statistics
    size_t get_memory_usage() const { return current_memory_bytes.load(); }
    size_t get_max_memory() const { return max_memory_bytes; }
    size_t get_entry_count() const;
    double get_hit_ratio() const;
    
    // Configuration methods
    void set_max_memory(size_t max_memory);
    void set_eviction_policy(EvictionPolicy policy);
    void set_read_ahead(bool enabled, size_t window = 5);
    void set_adaptive_sizing(bool enabled, double pressure_threshold = 0.8);
    
    // Performance statistics
    struct CacheStats {
        size_t current_memory;
        size_t max_memory;
        size_t entry_count;
        uint64_t hits;
        uint64_t misses;
        uint64_t evictions;
        uint64_t accesses;
        double hit_ratio;
    };
    
    CacheStats get_stats() const;
};