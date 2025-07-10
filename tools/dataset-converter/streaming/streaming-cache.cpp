#include "streaming-cache.h"

#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <sys/resource.h>

#include "../../common/log.h"
#include "llama-impl.h"

StreamingCache::StreamingCache(size_t max_memory)
    : max_memory_bytes(max_memory), 
      current_memory_bytes(0),
      read_ahead_enabled(false),
      read_ahead_window(5),
      hit_count(0),
      miss_count(0),
      eviction_count(0),
      access_count(0),
      timestamp_counter(0),
      eviction_policy(EvictionPolicy::ADAPTIVE),
      memory_pressure_threshold(0.8),
      adaptive_sizing_enabled(true),
      initial_max_memory(max_memory) {
}

StreamingCache::~StreamingCache() {
    clear();
}

void StreamingCache::evict_lru() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    while (current_memory_bytes > max_memory_bytes && !cache_list.empty()) {
        // Remove least recently used entry (back of list)
        auto& lru_entry = cache_list.back();

        LLAMA_LOG_DEBUG("Evicting sequence %zu from streaming cache (size: %zu bytes, LRU policy)",
                       lru_entry.sequence_id, lru_entry.size);

        // Free the data
        free(lru_entry.data);
        current_memory_bytes -= lru_entry.size;

        // Remove from map
        cache_map.erase(lru_entry.sequence_id);

        // Remove from list
        cache_list.pop_back();
        
        // Update stats
        eviction_count++;
    }
}

void StreamingCache::evict_lfu() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    if (cache_list.empty()) {
        return;
    }
    
    // Find entry with lowest access count
    auto min_it = std::min_element(cache_list.begin(), cache_list.end(), 
        [](const CacheEntry& a, const CacheEntry& b) {
            return a.access_count < b.access_count;
        });
    
    if (min_it != cache_list.end()) {
        LLAMA_LOG_DEBUG("Evicting sequence %zu from streaming cache (size: %zu bytes, LFU policy)",
                       min_it->sequence_id, min_it->size);
        
        // Free the data
        free(min_it->data);
        current_memory_bytes -= min_it->size;
        
        // Remove from map
        cache_map.erase(min_it->sequence_id);
        
        // Remove from list
        cache_list.erase(min_it);
        
        // Update stats
        eviction_count++;
    }
}

void StreamingCache::evict_adaptive() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    if (cache_list.empty()) {
        return;
    }
    
    // Use LRU for sequential access patterns, LFU for random access
    // Determine access pattern by analyzing timestamp differences
    bool sequential_pattern = true;
    uint64_t last_id = 0;
    uint64_t sequential_count = 0;
    
    for (const auto& entry : cache_list) {
        if (last_id > 0 && entry.sequence_id != last_id + 1) {
            sequential_pattern = false;
            break;
        }
        if (last_id > 0 && entry.sequence_id == last_id + 1) {
            sequential_count++;
        }
        last_id = entry.sequence_id;
    }
    
    // If we have at least 3 sequential accesses, consider it sequential pattern
    if (sequential_count >= 3) {
        sequential_pattern = true;
    }
    
    if (sequential_pattern) {
        // For sequential access, LRU works better
        evict_lru();
    } else {
        // For random access, LFU often works better
        evict_lfu();
    }
}

void StreamingCache::update_access_stats(std::list<CacheEntry>::iterator entry_it) {
    entry_it->last_access_time = get_current_timestamp();
    entry_it->access_count++;
}

void StreamingCache::check_memory_pressure() {
    if (!adaptive_sizing_enabled) {
        return;
    }
    
    // Get system memory info
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    
    // Check if we're using too much memory
    double memory_ratio = (double)current_memory_bytes / max_memory_bytes;
    
    if (memory_ratio > memory_pressure_threshold) {
        // Under memory pressure, reduce cache size
        size_t new_max = max_memory_bytes * 0.8;
        if (new_max >= initial_max_memory * 0.25) {  // Don't go below 25% of initial
            set_max_memory(new_max);
            LLAMA_LOG_DEBUG("Memory pressure detected, reducing cache to %zu bytes", new_max);
        }
    } else if (memory_ratio < memory_pressure_threshold * 0.5) {
        // Low memory pressure, can increase cache size
        size_t new_max = max_memory_bytes * 1.2;
        if (new_max <= initial_max_memory * 2.0) {  // Don't go above 200% of initial
            set_max_memory(new_max);
            LLAMA_LOG_DEBUG("Low memory pressure, increasing cache to %zu bytes", new_max);
        }
    }
}

void StreamingCache::prefetch_sequences(uint64_t current_id) {
    if (!read_ahead_enabled) {
        return;
    }
    
    // Clear previous prefetch queue
    prefetch_queue.clear();
    
    // Add next few sequences to prefetch queue
    for (uint64_t i = 1; i <= read_ahead_window; i++) {
        prefetch_queue.push_back(current_id + i);
    }
    
    // Note: actual prefetching happens outside this class
    // This just prepares the list of sequences to prefetch
}

uint64_t StreamingCache::get_current_timestamp() {
    return ++timestamp_counter;
}

void* StreamingCache::get(uint64_t sequence_id) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    access_count++;
    
    auto it = cache_map.find(sequence_id);
    if (it == cache_map.end()) {
        miss_count++;
        
        // Prepare prefetch for sequential access
        prefetch_sequences(sequence_id);
        
        return nullptr; // Not in cache
    }

    // Move to front (most recently used)
    auto list_it = it->second;
    cache_list.splice(cache_list.begin(), cache_list, list_it);
    
    // Update access statistics
    update_access_stats(list_it);
    hit_count++;

    return list_it->data;
}

void StreamingCache::put(uint64_t sequence_id, void* data, size_t size) {
    if (!data || size == 0) {
        return;
    }

    std::lock_guard<std::mutex> lock(cache_mutex);
    
    // Check if already in cache
    auto existing_it = cache_map.find(sequence_id);
    if (existing_it != cache_map.end()) {
        // Update existing entry
        auto list_it = existing_it->second;

        // Free old data
        free(list_it->data);
        current_memory_bytes -= list_it->size;

        // Update with new data
        list_it->data = data;
        list_it->size = size;
        current_memory_bytes += size;

        // Move to front and update stats
        cache_list.splice(cache_list.begin(), cache_list, list_it);
        update_access_stats(list_it);
    } else {
        // Add new entry
        current_memory_bytes += size;

        // Evict if necessary based on current policy
        switch (eviction_policy) {
            case EvictionPolicy::LRU:
                evict_lru();
                break;
            case EvictionPolicy::LFU:
                evict_lfu();
                break;
            case EvictionPolicy::ADAPTIVE:
                evict_adaptive();
                break;
        }

        // Add to front of list (most recently used)
        cache_list.emplace_front(data, size, sequence_id);
        auto it = cache_list.begin();
        it->last_access_time = get_current_timestamp();
        it->access_count = 1;
        cache_map[sequence_id] = it;
    }

    LLAMA_LOG_DEBUG("Cached sequence %zu in streaming cache (size: %zu bytes, total: %zu/%zu bytes)",
                   sequence_id, size, current_memory_bytes.load(), max_memory_bytes);
                   
    // Check memory pressure periodically
    if (access_count % 100 == 0) {
        check_memory_pressure();
    }
}

void StreamingCache::remove(uint64_t sequence_id) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    auto it = cache_map.find(sequence_id);
    if (it != cache_map.end()) {
        auto list_it = it->second;

        // Free the data
        free(list_it->data);
        current_memory_bytes -= list_it->size;

        // Remove from both containers
        cache_list.erase(list_it);
        cache_map.erase(it);

        LLAMA_LOG_DEBUG("Removed sequence %zu from streaming cache", sequence_id);
    }
}

void StreamingCache::clear() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    // Free all cached data
    for (auto& entry : cache_list) {
        free(entry.data);
    }

    cache_list.clear();
    cache_map.clear();
    current_memory_bytes = 0;
    
    // Reset statistics
    hit_count = 0;
    miss_count = 0;
    eviction_count = 0;
    access_count = 0;
    timestamp_counter = 0;

    LLAMA_LOG_DEBUG("Cleared streaming cache");
}

void StreamingCache::set_max_memory(size_t max_memory) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    max_memory_bytes = max_memory;

    // Evict if current usage exceeds new limit
    switch (eviction_policy) {
        case EvictionPolicy::LRU:
            evict_lru();
            break;
        case EvictionPolicy::LFU:
            evict_lfu();
            break;
        case EvictionPolicy::ADAPTIVE:
            evict_adaptive();
            break;
    }

    LLAMA_LOG_DEBUG("Set streaming cache max memory to %zu bytes", max_memory_bytes);
}

size_t StreamingCache::get_entry_count() const {
    // Can't use lock_guard with const mutex, so we'll just return the size
    // This is safe for reading the size of an atomic variable
    return cache_map.size();
}

double StreamingCache::get_hit_ratio() const {
    if (access_count == 0) {
        return 0.0;
    }
    return static_cast<double>(hit_count) / access_count;
}

void StreamingCache::set_eviction_policy(EvictionPolicy policy) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    eviction_policy = policy;
    LLAMA_LOG_DEBUG("Set streaming cache eviction policy to %d", static_cast<int>(policy));
}

void StreamingCache::set_read_ahead(bool enabled, size_t window) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    read_ahead_enabled = enabled;
    read_ahead_window = window;
    LLAMA_LOG_DEBUG("Set streaming cache read-ahead to %s with window %zu", 
                   enabled ? "enabled" : "disabled", window);
}

void StreamingCache::set_adaptive_sizing(bool enabled, double pressure_threshold) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    adaptive_sizing_enabled = enabled;
    memory_pressure_threshold = pressure_threshold;
    LLAMA_LOG_DEBUG("Set streaming cache adaptive sizing to %s with threshold %.2f", 
                   enabled ? "enabled" : "disabled", pressure_threshold);
}

StreamingCache::CacheStats StreamingCache::get_stats() const {
    // Can't use lock_guard with const mutex in a const method
    // We'll create a copy of the stats without locking
    // This is not thread-safe but acceptable for stats
    CacheStats stats;
    stats.current_memory = current_memory_bytes.load();
    stats.max_memory = max_memory_bytes;
    stats.entry_count = cache_map.size();
    stats.hits = hit_count;
    stats.misses = miss_count;
    stats.evictions = eviction_count;
    stats.accesses = access_count;
    stats.hit_ratio = (access_count > 0) ? static_cast<double>(hit_count) / access_count : 0.0;
    
    return stats;
}
