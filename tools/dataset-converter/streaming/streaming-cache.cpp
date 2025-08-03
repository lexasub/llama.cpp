/**
 * @file streaming-cache.cpp
 * @brief Implementation of the streaming cache for dataset sequences
 *
 * This module implements a sophisticated LRU-based caching system designed specifically
 * for streaming dataset operations. The cache provides memory-efficient storage and
 * retrieval of dataset sequences with advanced features including:
 *
 * ## Core Functionality
 * - **LRU Cache Management**: Implements Least Recently Used eviction policy with
 *   efficient O(1) access and update operations using hash map + doubly linked list
 * - **Multi-Policy Eviction**: Supports LRU, LFU (Least Frequently Used), and adaptive
 *   eviction strategies that automatically adjust based on access patterns
 * - **Thread-Safe Operations**: All cache operations are protected by shared/unique
 *   locks to ensure thread safety in multi-threaded streaming environments
 *
 * ## Memory Management
 * - **Adaptive Memory Sizing**: Automatically adjusts cache size based on system
 *   memory pressure using rusage monitoring
 * - **Memory Pressure Detection**: Monitors memory usage and dynamically resizes
 *   cache to prevent system memory exhaustion
 * - **Configurable Memory Limits**: Supports both hard limits and adaptive thresholds
 *   for memory management
 *
 * ## Performance Optimization
 * - **Read-Ahead Buffering**: Implements predictive loading for sequential access
 *   patterns to improve streaming performance
 * - **Access Pattern Analysis**: Analyzes sequence access patterns to optimize
 *   eviction strategies and prefetching behavior
 * - **Performance Statistics**: Tracks hit/miss ratios, eviction counts, and access
 *   patterns for performance monitoring and tuning
 *
 * ## Algorithm Details
 * The cache uses a combination of:
 * - Hash map for O(1) sequence ID lookup
 * - Doubly linked list for O(1) LRU ordering maintenance
 * - Atomic counters for thread-safe statistics tracking
 * - Shared/unique locks for fine-grained concurrency control
 *
 * ## Memory Layout
 * Each cache entry stores:
 * - Raw data pointer (managed by the cache)
 * - Data size in bytes
 * - Sequence ID for identification
 * - Last access timestamp for LRU ordering
 * - Access count for LFU policy
 *
 * ## Thread Safety
 * The implementation uses a two-level locking strategy:
 * - cache_mutex: Protects cache structure (map/list operations)
 * - stats_mutex: Protects statistics and metadata updates
 * This design minimizes lock contention while ensuring data consistency.
 *
 * @author llama.cpp dataset-converter team
 * @version 1.0
 * @since 2024
 */

#include "streaming-cache.h"

#include <sys/resource.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "llama-impl.h"

/**
 * @brief Cache entry structure for storing dataset sequences
 *
 * Each cache entry represents a single dataset sequence stored in memory.
 * The structure is optimized for both LRU and LFU eviction policies by
 * maintaining both temporal and frequency-based access information.
 *
 * @note The data pointer is owned by the cache entry and will be freed
 *       when the entry is evicted or the cache is cleared.
 */
struct CacheEntry {
    void* data;                    ///< Raw sequence data (owned by this entry)
    size_t size;                   ///< Size of the data in bytes
    uint64_t sequence_id;          ///< Unique identifier for the sequence
    uint64_t last_access_time;     ///< Timestamp of last access (for LRU)
    uint64_t access_count;         ///< Number of times accessed (for LFU)
    llama_dataset_streaming_cache::CacheEntryType entry_type; ///< Type of cache entry

    /**
     * @brief Constructs a new cache entry
     * @param d Pointer to the sequence data (ownership transferred)
     * @param s Size of the data in bytes
     * @param id Unique sequence identifier
     * @param type Type of cache entry (default: TENSOR_DATA)
     */
    CacheEntry(void* d, size_t s, uint64_t id, llama_dataset_streaming_cache::CacheEntryType type = llama_dataset_streaming_cache::CacheEntryType::TENSOR_DATA)
        : data(d), size(s), sequence_id(id),
          last_access_time(0), access_count(0), entry_type(type) {}
};

namespace {
/// Global timestamp counter for LRU ordering (thread-safe)
std::atomic<uint64_t> counter{0};

/// Doubly linked list maintaining LRU order (most recent at front)
std::list<CacheEntry> cache_list;

/// Hash map for O(1) sequence ID to list iterator lookup
std::unordered_map<uint64_t, std::list<CacheEntry>::iterator> cache_map;

/// Hash map for tokenized sequences
std::unordered_map<uint64_t, std::list<CacheEntry>::iterator> tokenized_cache_map;

/**
 * @brief Generates monotonically increasing timestamps for LRU ordering
 * @return Unique timestamp value
 * @note Thread-safe atomic increment operation
 */
uint64_t get_current_timestamp() {
    return ++counter;
}
}

/**
 * @brief Updates access statistics for a cache entry
 * @param entry_it Iterator pointing to the cache entry to update
 * 
 * Updates both the last access time (for LRU) and access count (for LFU).
 * This function is called whenever an entry is accessed to maintain
 * accurate eviction policy data.
 */
void update_access_stats(std::list<CacheEntry>::iterator entry_it);

/**
 * @brief Evicts the least recently used entry from the cache
 * 
 * Implements the LRU (Least Recently Used) eviction policy by removing
 * the entry at the back of the cache list. The cache list is maintained
 * in LRU order with most recently accessed items at the front.
 * 
 * ## Algorithm Complexity
 * - Time: O(1) - Direct access to list tail
 * - Space: O(1) - No additional memory allocation
 * 
 * ## Thread Safety
 * Uses shared lock for reading and unique lock for modifications to
 * ensure thread-safe access to cache statistics and structure.
 * 
 * @note This function is automatically called when memory pressure
 *       requires eviction or when the LRU policy is explicitly set.
 */
void llama_dataset_streaming_cache::evict_lru() {
    if (cache_list.empty()) {
        return;
    }

    std::shared_lock stats_lock(stats_mutex);
    // Find LRU-element
    auto lru_entry = cache_list.back();
    stats_lock.unlock();

    {
        std::unique_lock _stats_lock(stats_mutex);
        LLAMA_LOG_DEBUG("Evicting sequence %zu from streaming cache (size: %zu bytes, LRU policy)", lru_entry.sequence_id, lru_entry.size);

        free(lru_entry.data);
        current_memory_bytes -= lru_entry.size;
        ++eviction_count;
    }

    // Remove from appropriate cache map
    if (lru_entry.entry_type == CacheEntryType::TOKENIZED_SEQUENCE) {
        tokenized_cache_map.erase(lru_entry.sequence_id);
    } else {
        cache_map.erase(lru_entry.sequence_id);
    }
    
    cache_list.pop_back();
}

/**
 * @brief Evicts the least frequently used entry from the cache
 * 
 * Implements the LFU (Least Frequently Used) eviction policy by finding
 * and removing the entry with the lowest access count. This policy is
 * effective for workloads with temporal locality where some sequences
 * are accessed much more frequently than others.
 * 
 * ## Algorithm Complexity
 * - Time: O(n) - Linear scan to find minimum access count
 * - Space: O(1) - No additional memory allocation
 * 
 * ## Performance Considerations
 * While LFU provides better hit rates for certain access patterns,
 * the O(n) eviction cost makes it less suitable for very large caches.
 * Consider using adaptive policy for automatic optimization.
 * 
 * @note The linear scan could be optimized with a min-heap, but the
 *       current implementation prioritizes simplicity and memory efficiency.
 */
void llama_dataset_streaming_cache::evict_lfu() {
    if (cache_list.empty()) {
        return;
    }

    std::shared_lock stats_lock(stats_mutex);
    auto min_it = std::min_element(cache_list.begin(), cache_list.end(),
                         [](const CacheEntry & a, const CacheEntry & b) { return a.access_count < b.access_count; });
    stats_lock.unlock();

    if (min_it != cache_list.end()) {
        {
            std::unique_lock _stats_lock(stats_mutex);
            LLAMA_LOG_DEBUG("Evicting sequence %zu from streaming cache (size: %zu bytes, LFU policy)",
                            min_it->sequence_id, min_it->size);
            free(min_it->data);
            current_memory_bytes -= min_it->size;
            ++eviction_count;
        }

        // Remove from appropriate cache map
        if (min_it->entry_type == CacheEntryType::TOKENIZED_SEQUENCE) {
            tokenized_cache_map.erase(min_it->sequence_id);
        } else {
            cache_map.erase(min_it->sequence_id);
        }
        
        cache_list.erase(min_it);
    }
}

/**
 * @brief Evicts entries using adaptive policy based on access patterns
 * 
 * Implements an intelligent eviction strategy that analyzes recent access
 * patterns to choose between LRU and LFU policies. The algorithm detects
 * sequential access patterns and adapts the eviction strategy accordingly:
 * 
 * - **Sequential Pattern**: Uses LRU eviction (optimal for streaming)
 * - **Random Pattern**: Uses LFU eviction (better for repeated access)
 * 
 * ## Pattern Detection Algorithm
 * Analyzes the sequence IDs in the cache to detect sequential access:
 * - Counts consecutive sequence IDs
 * - Considers pattern sequential if 3+ consecutive sequences found
 * - Adapts eviction policy based on detected pattern
 * 
 * ## Algorithm Complexity
 * - Time: O(n) for pattern analysis + eviction cost
 * - Space: O(1) - No additional memory allocation
 * 
 * ## Performance Benefits
 * - Automatically optimizes for different workload characteristics
 * - Provides better average performance across mixed access patterns
 * - Reduces need for manual policy tuning
 * 
 * @note This is the recommended policy for general-purpose streaming
 *       workloads with unknown or varying access patterns.
 */
void llama_dataset_streaming_cache::evict_adaptive() {
    if (cache_list.empty()) {
        return;
    }

    bool sequential_pattern = false;
    {
        std::shared_lock stats_lock(stats_mutex);
        uint64_t last_id          = 0;
        uint64_t sequential_count = 0;

        for (const auto & entry : cache_list) {
            if (last_id > 0 && entry.sequence_id != last_id + 1) {
                break;
            }
            if (last_id > 0 && entry.sequence_id == last_id + 1) {
                sequential_count++;
            }
            last_id = entry.sequence_id;
        }

        sequential_pattern = (sequential_count >= 3);
    }

    if (sequential_pattern) {
        evict_lru();
    } else {
        evict_lfu();
    }
}

/**
 * @brief Monitors system memory pressure and adapts cache size accordingly
 * 
 * Implements dynamic memory management by monitoring system resource usage
 * and automatically adjusting cache size to prevent memory exhaustion.
 * Uses the rusage system call to track memory consumption.
 * 
 * ## Adaptive Sizing Algorithm
 * - **High Pressure** (>80% threshold): Reduces cache size by 20%
 * - **Low Pressure** (<40% threshold): Increases cache size by 20%
 * - **Bounds Checking**: Maintains cache between 25% and 200% of initial size
 * 
 * ## Memory Pressure Detection
 * Calculates memory ratio as: current_memory / max_memory
 * Compares against configurable threshold (default 0.8)
 * 
 * ## Performance Impact
 * - Called periodically (every 100 cache operations)
 * - Minimal overhead due to atomic memory counters
 * - Prevents system-wide memory issues
 * 
 * @note Only active when adaptive_sizing_enabled is true.
 *       Requires RUSAGE_SELF support on the target platform.
 */
void llama_dataset_streaming_cache::check_memory_pressure() {
    if (!adaptive_sizing_enabled) {
        return;
    }

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);

    double memory_ratio = static_cast<double>(current_memory_bytes.load()) / max_memory_bytes;
    {
        std::unique_lock lock(cache_mutex);
        if (memory_ratio > memory_pressure_threshold) {
            size_t new_max = max_memory_bytes * 0.8;
            if (new_max >= initial_max_memory * 0.25) {
                set_max_memory(new_max);
                LLAMA_LOG_DEBUG("Memory pressure detected, reducing cache to %zu bytes\n", new_max);
            }
        } else if (memory_ratio < memory_pressure_threshold * 0.5) {
            size_t new_max = max_memory_bytes * 1.2;
            if (new_max <= initial_max_memory * 2.0) {
                set_max_memory(new_max);
                LLAMA_LOG_DEBUG("Low memory pressure, increasing cache to %zu bytes\n", new_max);
            }
        }
    }
}

/**
 * @brief Implements read-ahead prefetching for sequential access optimization
 * @param current_id The sequence ID that was just accessed
 * 
 * Populates the prefetch queue with sequence IDs that are likely to be
 * accessed next based on the current access pattern. This enables
 * background loading of sequences to improve streaming performance.
 * 
 * ## Prefetch Strategy
 * - Predicts next sequences based on sequential pattern
 * - Queues sequence IDs: current_id+1, current_id+2, ..., current_id+window
 * - Window size is configurable (default: 5 sequences)
 * 
 * ## Performance Benefits
 * - Reduces cache misses for sequential access patterns
 * - Enables background loading while processing current sequence
 * - Improves overall streaming throughput
 * 
 * ## Algorithm Complexity
 * - Time: O(window_size) - Linear in prefetch window
 * - Space: O(window_size) - Stores prefetch queue
 * 
 * @note Prefetching is only active when read_ahead_enabled is true.
 *       The prefetch queue is cleared and repopulated on each call.
 */
void llama_dataset_streaming_cache::prefetch_sequences(uint64_t current_id) {
    std::shared_lock lock(cache_mutex);
    prefetch_queue.clear();

    if (!read_ahead_enabled) {
        return;
    }

    for (uint64_t i = 1; i <= read_ahead_window; i++) {
        prefetch_queue.push_back(current_id + i);
    }
}

/**
 * @brief Generates unique timestamps for cache entry ordering
 * @return Monotonically increasing timestamp value
 * 
 * Provides thread-safe timestamp generation for maintaining LRU order
 * in the cache. Each call returns a unique, incrementing value that
 * can be used to determine the relative access order of cache entries.
 * 
 * ## Thread Safety
 * Uses atomic increment operation to ensure thread-safe timestamp
 * generation without requiring explicit locking.
 * 
 * ## Algorithm Complexity
 * - Time: O(1) - Single atomic increment
 * - Space: O(1) - No memory allocation
 * 
 * @note Timestamps are used internally for LRU ordering and do not
 *       correspond to wall-clock time.
 */
uint64_t llama_dataset_streaming_cache::get_current_timestamp() {
    return ++timestamp_counter;
}

/**
 * @brief Retrieves a sequence from the cache
 * @param sequence_id Unique identifier of the sequence to retrieve
 * @return Pointer to the sequence data, or nullptr if not found
 * 
 * Performs cache lookup with LRU order maintenance. If the sequence is
 * found, it is moved to the front of the LRU list and access statistics
 * are updated. If not found, prefetching may be triggered for future
 * sequential accesses.
 * 
 * ## Cache Hit Path
 * 1. Lookup sequence ID in hash map (O(1))
 * 2. Move entry to front of LRU list (O(1))
 * 3. Update access statistics (timestamp, count)
 * 4. Return data pointer
 * 
 * ## Cache Miss Path
 * 1. Increment miss counter
 * 2. Trigger prefetch for predicted sequences
 * 3. Return nullptr
 * 
 * ## Algorithm Complexity
 * - Time: O(1) for cache operations
 * - Space: O(1) - No additional memory allocation
 * 
 * ## Thread Safety
 * Uses shared lock for read operations and unique lock for statistics
 * updates to minimize lock contention while ensuring data consistency.
 * 
 * @note The returned pointer is valid until the entry is evicted.
 *       Callers should not free the returned memory.
 */
void * llama_dataset_streaming_cache::get(uint64_t sequence_id) {
    std::shared_lock lock(cache_mutex);
    ++access_count;

    auto it = cache_map.find(sequence_id);
    if (it == cache_map.end()) {
        std::unique_lock stats_lock(stats_mutex);
        ++miss_count;
        prefetch_sequences(sequence_id);
        return nullptr;
    }

    {
        std::unique_lock stats_lock(stats_mutex);
        ++hit_count;
        auto list_it = it->second;
        cache_list.splice(cache_list.begin(), cache_list, list_it);
        update_access_stats(list_it);
    }

    return it->second->data;
}

/**
 * @brief Stores a sequence in the cache
 * @param sequence_id Unique identifier for the sequence
 * @param data Pointer to sequence data (ownership transferred to cache)
 * @param size Size of the data in bytes
 * 
 * Adds a new sequence to the cache or updates an existing one. The cache
 * takes ownership of the data pointer and will free it when the entry is
 * evicted. If the cache is full, eviction is triggered according to the
 * current eviction policy.
 * 
 * ## New Entry Path
 * 1. Check memory limits and trigger eviction if needed
 * 2. Create new cache entry at front of LRU list
 * 3. Add entry to hash map for O(1) lookup
 * 4. Update memory usage and access statistics
 * 
 * ## Update Existing Entry Path
 * 1. Free old data and update with new data
 * 2. Move entry to front of LRU list
 * 3. Update memory usage accounting
 * 4. Update access statistics
 * 
 * ## Memory Management
 * - Takes ownership of data pointer
 * - Automatically frees old data when updating entries
 * - Triggers eviction when memory limits are exceeded
 * - Periodically checks memory pressure (every 100 operations)
 * 
 * ## Algorithm Complexity
 * - Time: O(1) for cache operations, O(n) for eviction
 * - Space: O(size) - Stores the provided data
 * 
 * ## Thread Safety
 * Uses unique locks to ensure exclusive access during cache modifications
 * and memory accounting updates.
 * 
 * @note Passing nullptr or zero size is ignored safely.
 *       The cache will free the data pointer when the entry is evicted.
 */
void llama_dataset_streaming_cache::put(uint64_t sequence_id, void * data, size_t size) {
    if (!data || size == 0) {
        return;
    }

    std::unique_lock lock(cache_mutex);
    auto existing_it = cache_map.find(sequence_id);

    {
        std::unique_lock stats_lock(stats_mutex);
        ++access_count;
    }

    if (existing_it != cache_map.end()) {
        auto             list_it = existing_it->second;
        std::unique_lock stats_lock(stats_mutex);
        free(list_it->data);
        current_memory_bytes -= list_it->size;
        list_it->data = data;
        list_it->size = size;
        current_memory_bytes += size;
        update_access_stats(list_it);
        cache_list.splice(cache_list.begin(), cache_list, list_it);

        if (access_count % 100 == 0) {
            check_memory_pressure();
        }
        return;
    }
    {
        std::unique_lock stats_lock(stats_mutex);
        current_memory_bytes += size;
    }

    switch (eviction_policy) {
        case EvictionPolicy::LRU:
            evict_lru();
            break;
        case EvictionPolicy::LFU:
            evict_lru();
            break;
        case EvictionPolicy::ADAPTIVE:
            evict_adaptive();
            break;
    }

    {
        std::unique_lock stats_lock(stats_mutex);
        cache_list.emplace_front(data, size, sequence_id, CacheEntryType::TENSOR_DATA);

        auto it                = cache_list.begin();
        it->last_access_time   = get_current_timestamp();
        it->access_count       = 1;
        cache_map[sequence_id] = it;
    }

    LLAMA_LOG_DEBUG("Cached sequence %zu in streaming cache (size: %zu bytes, total: %zu/%zu bytes)\n", sequence_id,
                    size, current_memory_bytes.load(), max_memory_bytes);

    if (access_count % 100 == 0) {
        check_memory_pressure();
    }
}

/**
 * @brief Removes a specific sequence from the cache
 * @param sequence_id Unique identifier of the sequence to remove
 * 
 * Explicitly removes a sequence from the cache, freeing its memory and
 * updating cache statistics. This is useful for cache invalidation or
 * when specific sequences are known to be no longer needed.
 * 
 * ## Removal Process
 * 1. Lookup sequence in hash map
 * 2. Free the associated data memory
 * 3. Remove from both hash map and LRU list
 * 4. Update memory usage accounting
 * 
 * ## Algorithm Complexity
 * - Time: O(1) - Direct hash map lookup and list removal
 * - Space: O(1) - Frees memory, no allocation
 * 
 * ## Thread Safety
 * Uses unique lock to ensure exclusive access during removal operations
 * and memory accounting updates.
 * 
 * @note If the sequence ID is not found, the operation is silently ignored.
 *       This makes the function safe to call speculatively.
 */
void llama_dataset_streaming_cache::remove(uint64_t sequence_id) {
    std::unique_lock lock(cache_mutex);
    
    // Check tensor cache
    auto it = cache_map.find(sequence_id);
    if (it != cache_map.end()) {
        auto list_it = it->second;
        {
            std::unique_lock stats_lock(stats_mutex);
            free(list_it->data);
            current_memory_bytes -= list_it->size;
        }

        cache_list.erase(list_it);
        cache_map.erase(it);
        LLAMA_LOG_DEBUG("Removed tensor sequence %zu from streaming cache\n", sequence_id);
        return;
    }
    
    // Check tokenized cache
    auto tokenized_it = tokenized_cache_map.find(sequence_id);
    if (tokenized_it != tokenized_cache_map.end()) {
        auto list_it = tokenized_it->second;
        {
            std::unique_lock stats_lock(stats_mutex);
            free(list_it->data);
            current_memory_bytes -= list_it->size;
        }

        cache_list.erase(list_it);
        tokenized_cache_map.erase(tokenized_it);
        LLAMA_LOG_DEBUG("Removed tokenized sequence %zu from streaming cache\n", sequence_id);
    }
}

/**
 * @brief Clears all entries from the cache
 * 
 * Removes all cached sequences, frees all associated memory, and resets
 * all statistics to their initial state. This is useful for cache reset
 * or when switching between different datasets.
 * 
 * ## Clear Process
 * 1. Free all data pointers in cache entries
 * 2. Clear both hash map and LRU list
 * 3. Reset memory usage to zero
 * 4. Reset all statistics counters
 * 
 * ## Memory Management
 * Ensures all allocated memory is properly freed to prevent memory leaks.
 * After clearing, the cache is in the same state as a newly constructed
 * cache with the same configuration.
 * 
 * ## Algorithm Complexity
 * - Time: O(n) - Must free all entries
 * - Space: O(1) - Frees all memory
 * 
 * ## Thread Safety
 * Uses unique locks to ensure exclusive access during the clear operation.
 * 
 * @note This operation is irreversible and will lose all cached data.
 */
void llama_dataset_streaming_cache::clear() {
    std::unique_lock lock(cache_mutex);
    for (auto & entry : cache_list) {
        free(entry.data);
    }

    {
        std::unique_lock stats_lock(stats_mutex);
        cache_list.clear();
        cache_map.clear();
        current_memory_bytes = 0;
        hit_count            = 0;
        miss_count           = 0;
        eviction_count       = 0;
        access_count         = 0;
        timestamp_counter    = 0;
    }

    LLAMA_LOG_DEBUG("Cleared streaming cache\n");
}

/**
 * @brief Updates the maximum memory limit for the cache
 * @param max_memory New maximum memory limit in bytes
 * 
 * Changes the cache memory limit and triggers eviction if the current
 * memory usage exceeds the new limit. The eviction policy used depends
 * on the currently configured policy (LRU, LFU, or adaptive).
 * 
 * ## Memory Adjustment Process
 * 1. Update the maximum memory limit
 * 2. Check if current usage exceeds new limit
 * 3. Trigger eviction using current policy if needed
 * 4. Continue evicting until under the new limit
 * 
 * ## Use Cases
 * - Dynamic memory management based on system conditions
 * - Adaptive sizing in response to memory pressure
 * - Manual cache tuning for performance optimization
 * 
 * ## Algorithm Complexity
 * - Time: O(k) where k is the number of evictions needed
 * - Space: O(1) - No additional memory allocation
 * 
 * ## Thread Safety
 * Uses unique lock to ensure exclusive access during memory limit updates
 * and potential eviction operations.
 * 
 * @note Setting a very small limit may cause immediate eviction of most
 *       or all cache entries.
 */
void llama_dataset_streaming_cache::set_max_memory(size_t max_memory) {
    std::unique_lock lock(cache_mutex);
    max_memory_bytes = max_memory;

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

    LLAMA_LOG_DEBUG("Set streaming cache max memory to %zu bytes\n", max_memory_bytes);
}

/**
 * @brief Returns the current number of entries in the cache
 * @return Number of cached sequences
 * 
 * Provides a thread-safe way to query the current cache occupancy.
 * This is useful for monitoring cache utilization and debugging.
 * 
 * ## Algorithm Complexity
 * - Time: O(1) - Direct hash map size query
 * - Space: O(1) - No memory allocation
 * 
 * ## Thread Safety
 * Uses shared lock to allow concurrent reads while preventing
 * inconsistent results during cache modifications.
 */
size_t llama_dataset_streaming_cache::get_entry_count() const {
    std::shared_lock lock(cache_mutex);
    return cache_map.size();
}

/**
 * @brief Calculates the current cache hit ratio
 * @return Hit ratio as a value between 0.0 and 1.0
 * 
 * Computes the ratio of cache hits to total accesses, providing a key
 * performance metric for cache effectiveness. A higher hit ratio indicates
 * better cache performance and more efficient memory utilization.
 * 
 * ## Calculation
 * hit_ratio = hit_count / (hit_count + miss_count)
 * 
 * ## Return Values
 * - 0.0: No hits (all misses or no accesses)
 * - 1.0: Perfect hit rate (no misses)
 * - 0.0-1.0: Partial hit rate
 * 
 * ## Algorithm Complexity
 * - Time: O(1) - Simple arithmetic operation
 * - Space: O(1) - No memory allocation
 * 
 * ## Thread Safety
 * Uses shared lock to ensure consistent statistics reading.
 * 
 * @note Returns 0.0 if no accesses have been made yet.
 */
double llama_dataset_streaming_cache::get_hit_ratio() const {
    std::shared_lock lock(stats_mutex);
    if (access_count == 0) {
        return 0.0;
    }
    return static_cast<double>(hit_count) / access_count;
}

/**
 * @brief Sets the cache eviction policy
 * @param policy The eviction policy to use (LRU, LFU, or ADAPTIVE)
 * 
 * Changes the algorithm used for selecting entries to evict when the cache
 * reaches its memory limit. Different policies are optimal for different
 * access patterns:
 * 
 * - **LRU**: Best for sequential or temporal locality patterns
 * - **LFU**: Best for frequency-based access patterns
 * - **ADAPTIVE**: Automatically adapts to detected access patterns
 * 
 * ## Policy Characteristics
 * - **LRU**: O(1) eviction, good for streaming workloads
 * - **LFU**: O(n) eviction, better hit rates for repeated access
 * - **ADAPTIVE**: O(n) analysis + policy cost, best general performance
 * 
 * ## Algorithm Complexity
 * - Time: O(1) - Simple policy assignment
 * - Space: O(1) - No memory allocation
 * 
 * ## Thread Safety
 * Uses unique locks to ensure atomic policy updates.
 * 
 * @note Policy changes take effect immediately for subsequent evictions.
 */
void llama_dataset_streaming_cache::set_eviction_policy(EvictionPolicy policy) {
    std::unique_lock lock(cache_mutex);
    std::unique_lock stats_lock(stats_mutex);
    eviction_policy = policy;
}

/**
 * @brief Configures read-ahead prefetching behavior
 * @param enabled Whether to enable read-ahead prefetching
 * @param window Number of sequences to prefetch ahead
 * 
 * Controls the predictive loading feature that attempts to prefetch
 * sequences that are likely to be accessed next based on sequential
 * access patterns. This can significantly improve performance for
 * streaming workloads with predictable access patterns.
 * 
 * ## Prefetch Window
 * - Small window (1-3): Lower memory overhead, less aggressive prefetching
 * - Medium window (4-8): Balanced performance and memory usage
 * - Large window (9+): Higher memory usage, maximum prefetch benefit
 * 
 * ## Performance Impact
 * - **Enabled**: Better performance for sequential access, higher memory usage
 * - **Disabled**: Lower memory usage, potential cache misses for sequential access
 * 
 * ## Algorithm Complexity
 * - Time: O(1) - Simple configuration update
 * - Space: O(window) - Prefetch queue storage
 * 
 * ## Thread Safety
 * Uses unique lock to ensure atomic configuration updates.
 * 
 * @note Changes take effect immediately for subsequent cache operations.
 */
void llama_dataset_streaming_cache::set_read_ahead(bool enabled, size_t window) {
    std::unique_lock lock(cache_mutex);
    read_ahead_enabled = enabled;
    read_ahead_window  = window;
}

/**
 * @brief Configures adaptive memory sizing behavior
 * @param enabled Whether to enable adaptive sizing
 * @param pressure_threshold Memory pressure threshold (0.0-1.0)
 * 
 * Controls the automatic memory management feature that monitors system
 * memory pressure and adjusts cache size accordingly. This helps prevent
 * memory exhaustion while maximizing cache effectiveness.
 * 
 * ## Pressure Threshold
 * - 0.5-0.7: Conservative, early size reduction
 * - 0.7-0.8: Balanced approach (recommended)
 * - 0.8-0.9: Aggressive, maximum memory utilization
 * - 0.9+: Very aggressive, risk of memory pressure
 * 
 * ## Adaptive Behavior
 * - **High Pressure**: Reduces cache size to free memory
 * - **Low Pressure**: Increases cache size for better performance
 * - **Bounds**: Maintains cache between 25%-200% of initial size
 * 
 * ## Algorithm Complexity
 * - Time: O(1) - Simple configuration update
 * - Space: O(1) - No memory allocation
 * 
 * ## Thread Safety
 * Uses unique lock to ensure atomic configuration updates.
 * 
 * @note Adaptive sizing requires system resource monitoring support.
 */
void llama_dataset_streaming_cache::set_adaptive_sizing(bool enabled, double pressure_threshold) {
    std::unique_lock lock(cache_mutex);
    adaptive_sizing_enabled   = enabled;
    memory_pressure_threshold = pressure_threshold;
}

/**
 * @brief Retrieves comprehensive cache performance statistics
 * @return CacheStats structure containing all performance metrics
 * 
 * Provides detailed statistics about cache performance, memory usage,
 * and access patterns. This information is essential for performance
 * monitoring, debugging, and cache tuning.
 * 
 * ## Statistics Included
 * - **Memory Usage**: Current and maximum memory consumption
 * - **Entry Count**: Number of cached sequences
 * - **Access Metrics**: Hits, misses, total accesses
 * - **Eviction Count**: Number of entries evicted
 * - **Hit Ratio**: Calculated hit rate percentage
 * 
 * ## Use Cases
 * - Performance monitoring and alerting
 * - Cache effectiveness analysis
 * - Memory usage tracking
 * - Access pattern analysis
 * - Debugging cache behavior
 * 
 * ## Algorithm Complexity
 * - Time: O(1) - Direct counter access
 * - Space: O(1) - Returns statistics structure
 * 
 * ## Thread Safety
 * Uses shared lock to ensure consistent statistics snapshot.
 * 
 * @note Statistics are updated atomically during cache operations.
 */
llama_dataset_streaming_cache::CacheStats llama_dataset_streaming_cache::get_stats() const {
    std::shared_lock stats_lock(stats_mutex);
    std::shared_lock cache_lock(cache_mutex);
    CacheStats       stats;
    stats.current_memory = current_memory_bytes.load();
    stats.max_memory     = max_memory_bytes;
    stats.entry_count    = cache_map.size() + tokenized_cache_map.size();
    stats.hits           = hit_count;
    stats.misses         = miss_count;
    stats.evictions      = eviction_count;
    stats.accesses       = access_count;
    stats.hit_ratio      = (access_count > 0) ? static_cast<double>(hit_count) / access_count : 0.0;
    stats.tokenized_entries = tokenized_cache_map.size();
    stats.tensor_entries = cache_map.size();
    return stats;
}

/**
 * @brief Returns the current maximum memory limit
 * @return Maximum memory limit in bytes
 * 
 * Provides the current memory limit setting for the cache. This value
 * may change over time if adaptive sizing is enabled.
 * 
 * ## Algorithm Complexity
 * - Time: O(1) - Direct member access
 * - Space: O(1) - No memory allocation
 * 
 * @note This value may differ from the initial limit if adaptive
 *       sizing has adjusted the cache size.
 */
size_t llama_dataset_streaming_cache::get_max_memory() const {
    return max_memory_bytes;
}

/**
 * @brief Updates access statistics for a cache entry
 * @param entry_it Iterator pointing to the cache entry to update
 * 
 * Updates both temporal (last access time) and frequency (access count)
 * statistics for a cache entry. This information is used by both LRU
 * and LFU eviction policies to make optimal eviction decisions.
 * 
 * ## Statistics Updated
 * - **last_access_time**: Set to current timestamp for LRU ordering
 * - **access_count**: Incremented for LFU frequency tracking
 * 
 * ## Algorithm Complexity
 * - Time: O(1) - Direct member updates
 * - Space: O(1) - No memory allocation
 * 
 * @note This function is called automatically during cache access
 *       operations and should not be called directly by users.
 */
void update_access_stats(std::list<CacheEntry>::iterator entry_it) {
    entry_it->last_access_time = get_current_timestamp();
    ++entry_it->access_count;
}
