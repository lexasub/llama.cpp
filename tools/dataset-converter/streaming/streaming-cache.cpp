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

struct CacheEntry {
    void* data;
    size_t size;
    uint64_t sequence_id;
    uint64_t last_access_time;
    uint64_t access_count;

    CacheEntry(void* d, size_t s, uint64_t id)
        : data(d), size(s), sequence_id(id),
          last_access_time(0), access_count(0) {}
};

namespace {
std::atomic<uint64_t> counter{0};
std::list<CacheEntry> cache_list;
std::unordered_map<uint64_t, std::list<CacheEntry>::iterator> cache_map;
uint64_t get_current_timestamp() {
    return ++counter;
}
}

void update_access_stats(std::list<CacheEntry>::iterator entry_it);
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

    cache_list.pop_back();
    cache_map.erase(lru_entry.sequence_id);
}

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

        cache_list.erase(min_it);
        cache_map.erase(min_it->sequence_id);
    }
}

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

uint64_t llama_dataset_streaming_cache::get_current_timestamp() {
    return ++timestamp_counter;
}

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
            evict_lfu();
            break;
        case EvictionPolicy::ADAPTIVE:
            evict_adaptive();
            break;
    }

    {
        std::unique_lock stats_lock(stats_mutex);
        cache_list.emplace_front(data, size, sequence_id);

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

void llama_dataset_streaming_cache::remove(uint64_t sequence_id) {
    std::unique_lock lock(cache_mutex);
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
        LLAMA_LOG_DEBUG("Removed sequence %zu from streaming cache\n", sequence_id);
    }
}

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

size_t llama_dataset_streaming_cache::get_entry_count() const {
    std::shared_lock lock(cache_mutex);
    return cache_map.size();
}

double llama_dataset_streaming_cache::get_hit_ratio() const {
    std::shared_lock lock(stats_mutex);
    if (access_count == 0) {
        return 0.0;
    }
    return static_cast<double>(hit_count) / access_count;
}

void llama_dataset_streaming_cache::set_eviction_policy(EvictionPolicy policy) {
    std::unique_lock lock(cache_mutex);
    std::unique_lock stats_lock(stats_mutex);
    eviction_policy = policy;
}

void llama_dataset_streaming_cache::set_read_ahead(bool enabled, size_t window) {
    std::unique_lock lock(cache_mutex);
    read_ahead_enabled = enabled;
    read_ahead_window  = window;
}

void llama_dataset_streaming_cache::set_adaptive_sizing(bool enabled, double pressure_threshold) {
    std::unique_lock lock(cache_mutex);
    adaptive_sizing_enabled   = enabled;
    memory_pressure_threshold = pressure_threshold;
}

llama_dataset_streaming_cache::CacheStats llama_dataset_streaming_cache::get_stats() const {
    std::shared_lock lock(stats_mutex);
    CacheStats       stats;
    stats.current_memory = current_memory_bytes.load();
    stats.max_memory     = max_memory_bytes;
    stats.entry_count    = cache_map.size();
    stats.hits           = hit_count;
    stats.misses         = miss_count;
    stats.evictions      = eviction_count;
    stats.accesses       = access_count;
    stats.hit_ratio      = (access_count > 0) ? static_cast<double>(hit_count) / access_count : 0.0;
    return stats;
}

size_t llama_dataset_streaming_cache::get_max_memory() const {
    return max_memory_bytes;
}

void update_access_stats(std::list<CacheEntry>::iterator entry_it) {
    entry_it->last_access_time = get_current_timestamp();
    ++entry_it->access_count;
}
