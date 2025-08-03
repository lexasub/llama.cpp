#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

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
#include <shared_mutex>
#include <cstring>
#include "platform/platform-compat.h"
#include "common/log.h"
#include "llama-impl.h"

class llama_dataset_streaming_cache {
public:
    enum class EvictionPolicy {
        LRU,
        LFU,
        ADAPTIVE
    };

    enum class CacheEntryType {
        TENSOR_DATA,
        TOKENIZED_SEQUENCE
    };

    struct CacheStats {
        size_t current_memory = 0;
        size_t max_memory = 0;
        size_t entry_count = 0;
        size_t hits = 0;
        size_t misses = 0;
        size_t evictions = 0;
        size_t accesses = 0;
        double hit_ratio = 0.0;
        size_t tokenized_entries = 0;
        size_t tensor_entries = 0;
    };

    explicit llama_dataset_streaming_cache(size_t max_memory)
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
          initial_max_memory(max_memory) {}

    ~llama_dataset_streaming_cache() {
        clear();
    }

    void evict_lru();

    void evict_lfu();

    void evict_adaptive();

    void check_memory_pressure();

    void prefetch_sequences(uint64_t current_id);

    uint64_t get_current_timestamp();

    void * get(uint64_t sequence_id);

    void put(uint64_t sequence_id, void * data, size_t size);

    // Tokenization-specific cache methods
    void * get_tokenized(uint64_t sequence_id);
    
    void put_tokenized(uint64_t sequence_id, const std::vector<int32_t> & tokens);
    
    bool has_tokenized(uint64_t sequence_id) const;

    void remove(uint64_t sequence_id);

    void clear();

    void set_max_memory(size_t max_memory);

    size_t get_entry_count() const;

    double get_hit_ratio() const;

    void set_eviction_policy(EvictionPolicy policy);

    void set_read_ahead(bool enabled, size_t window);

    void set_adaptive_sizing(bool enabled, double pressure_threshold);

    CacheStats get_stats() const;
    size_t get_max_memory() const;

  private:
    size_t max_memory_bytes;
    std::atomic<size_t> current_memory_bytes;
    bool read_ahead_enabled;
    size_t read_ahead_window;
    std::vector<uint64_t> prefetch_queue;

    mutable std::shared_mutex cache_mutex;
    mutable std::shared_mutex stats_mutex;

    std::atomic<uint64_t> hit_count;
    std::atomic<uint64_t> miss_count;
    std::atomic<uint64_t> eviction_count;
    std::atomic<uint64_t> access_count;
    std::atomic<uint64_t> timestamp_counter;

    EvictionPolicy eviction_policy;
    double memory_pressure_threshold;
    bool adaptive_sizing_enabled;
    size_t initial_max_memory;
};
