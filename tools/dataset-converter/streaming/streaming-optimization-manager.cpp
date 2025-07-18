#include "streaming-optimization-manager.h"

#include "../../common/log.h"
#include "llama-impl.h"

StreamingOptimizationManager::StreamingOptimizationManager(const std::string& name)
    : dataset_name(name),
      optimization_enabled(true),
      access_count(0),
      sequential_access_count(0),
      random_access_count(0),
      last_accessed_id(0) {
}

StreamingOptimizationManager::~StreamingOptimizationManager() {
    stop();
}

void StreamingOptimizationManager::initialize(size_t cache_size) {
    // Create components
    cache = std::make_unique<StreamingCache>(cache_size);
    read_ahead = std::make_unique<StreamingReadAhead>(5, 20);
    memory_monitor = std::make_unique<StreamingMemoryMonitor>(1000, 0.8, 0.6);

    // Set up callbacks
    memory_monitor->set_pressure_callback([this](double pressure) {
        handle_memory_pressure(pressure);
    });

    read_ahead->set_prefetch_callback([this](uint64_t sequence_id) {
        handle_prefetch(sequence_id);
    });

    LLAMA_LOG_DEBUG("Initialized streaming optimization manager for dataset '%s'", dataset_name.c_str());
}

void StreamingOptimizationManager::start() {
    if (!optimization_enabled) {
        LLAMA_LOG_DEBUG("Streaming optimization is disabled for dataset '%s'", dataset_name.c_str());
        return;
    }

    // Start components
    read_ahead->start();
    memory_monitor->start();

    LLAMA_LOG_DEBUG("Started streaming optimization for dataset '%s'", dataset_name.c_str());
}

void StreamingOptimizationManager::stop() {
    // Stop components
    if (read_ahead) {
        read_ahead->stop();
    }

    if (memory_monitor) {
        memory_monitor->stop();
    }

    LLAMA_LOG_DEBUG("Stopped streaming optimization for dataset '%s'", dataset_name.c_str());
}

void* StreamingOptimizationManager::get_sequence(uint64_t sequence_id) {
    if (!optimization_enabled || !cache) {
        return nullptr;
    }

    // Update access pattern statistics
    update_access_pattern(sequence_id);

    // Try to get from cache
    void* data = cache->get(sequence_id);

    // If not in cache, trigger prefetch for next sequences
    if (!data && read_ahead) {
        read_ahead->prefetch(sequence_id);
    }

    return data;
}

void StreamingOptimizationManager::put_sequence(uint64_t sequence_id, void* data, size_t size) {
    if (!optimization_enabled || !cache) {
        return;
    }

    cache->put(sequence_id, data, size);
}

void StreamingOptimizationManager::set_prefetch_callback(PrefetchCallback callback) {
    prefetch_callback = callback;
}

void StreamingOptimizationManager::set_cache_size(size_t size) {
    if (cache) {
        cache->set_max_memory(size);
    }
}

void StreamingOptimizationManager::set_read_ahead_window(size_t window) {
    if (read_ahead) {
        read_ahead->set_window_size(window);
    }
}

void StreamingOptimizationManager::set_memory_check_interval(size_t interval_ms) {
    if (memory_monitor) {
        memory_monitor->set_check_interval(interval_ms);
    }
}

void StreamingOptimizationManager::set_optimization_enabled(bool enabled) {
    optimization_enabled = enabled;

    if (enabled) {
        start();
    } else {
        stop();
    }
}

void StreamingOptimizationManager::set_read_ahead_enabled(bool enabled) {
    if (read_ahead) {
        if (enabled) {
            read_ahead->resume();
        } else {
            read_ahead->pause();
        }
    }
}

void StreamingOptimizationManager::set_adaptive_cache_enabled(bool enabled) {
    if (cache) {
        cache->set_adaptive_sizing(enabled, 0.8);
    }
}

StreamingOptimizationManager::OptimizationStats StreamingOptimizationManager::get_stats() const {
    OptimizationStats stats;

    if (cache) {
        stats.cache_stats = cache->get_stats();
    }

    if (read_ahead) {
        stats.read_ahead_status = read_ahead->get_status();
    }

    if (memory_monitor) {
        stats.memory_info = memory_monitor->get_memory_info();
    }

    stats.access_count = access_count;
    stats.sequential_access_count = sequential_access_count;
    stats.random_access_count = random_access_count;
    stats.sequential_access_ratio = (access_count > 0) ?
        static_cast<double>(sequential_access_count) / access_count : 0.0;

    return stats;
}

void StreamingOptimizationManager::handle_memory_pressure(double pressure) {
    if (!optimization_enabled || !cache) {
        return;
    }

    // Adjust cache size based on memory pressure
    if (pressure > 0.8) {
        // High pressure - reduce cache size
        size_t current_max = cache->get_max_memory();
        size_t new_max = current_max * 0.8;
        cache->set_max_memory(new_max);

        LLAMA_LOG_DEBUG("High memory pressure (%.2f) - reduced cache size to %zu bytes",
                       pressure, new_max);
    } else if (pressure < 0.6) {
        // Low pressure - increase cache size
        size_t current_max = cache->get_max_memory();
        size_t new_max = current_max * 1.2;
        cache->set_max_memory(new_max);

        LLAMA_LOG_DEBUG("Low memory pressure (%.2f) - increased cache size to %zu bytes",
                       pressure, new_max);
    }
}

void StreamingOptimizationManager::handle_prefetch(uint64_t sequence_id) {
    if (!optimization_enabled || !prefetch_callback || !cache) {
        return;
    }

    // Check if already in cache
    if (cache->get(sequence_id) != nullptr) {
        return;
    }

    // Prefetch the sequence
    size_t size = 0;
    void* data = prefetch_callback(sequence_id, &size);

    if (data && size > 0) {
        // Add to cache
        cache->put(sequence_id, data, size);
        LLAMA_LOG_DEBUG("Prefetched sequence %zu (size: %zu bytes)", sequence_id, size);
    }
}

void StreamingOptimizationManager::update_access_pattern(uint64_t sequence_id) {
    access_count++;

    // Check if this is sequential access
    if (last_accessed_id > 0 && sequence_id == last_accessed_id + 1) {
        sequential_access_count++;
    } else if (last_accessed_id > 0 && sequence_id != last_accessed_id) {
        random_access_count++;
    }

    last_accessed_id = sequence_id;

    // Adjust read-ahead window based on access pattern
    if (read_ahead && access_count % 100 == 0) {
        double sequential_ratio = static_cast<double>(sequential_access_count) / access_count;

        if (sequential_ratio > 0.8) {
            // Mostly sequential access - increase read-ahead window
            read_ahead->set_window_size(10);
        } else if (sequential_ratio < 0.2) {
            // Mostly random access - reduce read-ahead window
            read_ahead->set_window_size(2);
        } else {
            // Mixed access pattern - moderate read-ahead window
            read_ahead->set_window_size(5);
        }
    }
}
