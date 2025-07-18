#include <cstdlib>
#include <cstring>
#include <memory>

#include "llama-dataset-internal.h"
#include "llama-dataset-utils.h"
#include "llama-dataset.h"
#include "llama-impl.h"
#include "streaming-cache.h"
#include "streaming-optimization-manager.h"

static llama_dataset_stream_optimization_manager* llama_dataset_streaming_get_optimization_manager(struct llama_dataset* dataset) {
    if (!dataset || !dataset->streaming || !dataset->streaming_cache) {
        return nullptr;
    }

    // Check if we already have an optimization manager
    llama_dataset_stream_optimization_manager* manager = static_cast<llama_dataset_stream_optimization_manager*>(dataset->optimization_manager);

    // If not, create one and initialize it
    if (!manager) {
        // Create a name for the manager based on the dataset type
        std::string name;
        switch (dataset->type) {
            case DATASET_GGUF:
                name = "gguf-dataset";
                break;
            case DATASET_PARQUET:
                name = "parquet-dataset";
                break;
            case DATASET_TEXT:
                name = "text-dataset";
                break;
            default:
                name = "unknown-dataset";
                break;
        }

        // Create and initialize the manager
        manager = new llama_dataset_stream_optimization_manager(name);

        // Get the current cache size from the streaming cache
        llama_dataset_streaming_cache* cache = dataset->streaming_cache;
        size_t cache_size = cache ? cache->get_max_memory() : 64 * 1024 * 1024; // Default 64MB

        manager->initialize(cache_size);

        // Set up the prefetch callback to load sequences from the dataset
        manager->set_prefetch_callback([dataset](uint64_t seq_id, size_t* size_out) -> void* {
            // Check if the sequence is valid
            if (seq_id >= llama_dataset_n_sequences(dataset)) {
                return nullptr;
            }

            // Get the sequence length
            int32_t seq_len = llama_dataset_sequence_length(dataset, seq_id);
            if (seq_len <= 0) {
                return nullptr;
            }

            // Get the sequence data
            const int32_t* seq_data = llama_dataset_sequence(dataset, seq_id);
            if (!seq_data) {
                return nullptr;
            }

            // Allocate memory for the sequence
            size_t size = seq_len * sizeof(int32_t);
            void* data = malloc(size);
            if (!data) {
                return nullptr;
            }

            // Copy the sequence data
            memcpy(data, seq_data, size);

            // Set the output size
            if (size_out) {
                *size_out = size;
            }

            return data;
        });

        // Start the optimization manager
        manager->start();

        // Store the manager in the dataset
        dataset->optimization_manager = manager;
    }

    return manager;
}

/**
 * @brief Configure streaming cache size for a dataset.
 *
 * @param dataset Dataset to configure
 * @param cache_size_bytes Maximum cache size in bytes
 * @return true on success, false on error
 */
bool llama_dataset_set_streaming_cache_size(struct llama_dataset* dataset, size_t cache_size_bytes) {
    if (!dataset || !dataset->streaming) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Dataset is not in streaming mode");
        return false;
    }

    // Get the streaming cache
    llama_dataset_streaming_cache* cache = dataset->streaming_cache;
    if (!cache) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Streaming cache not initialized");
        return false;
    }

    // Set the cache size
    cache->set_max_memory(cache_size_bytes);

    // Update the optimization manager if it exists
    if (auto manager = llama_dataset_streaming_get_optimization_manager(dataset)) {
        manager->set_cache_size(cache_size_bytes);
    }

    LLAMA_LOG_INFO("Set streaming cache size to %zu bytes", cache_size_bytes);
    return true;
}

/**
 * @brief Enable or disable read-ahead buffering for streaming.
 *
 * @param dataset Dataset to configure
 * @param enabled Whether to enable read-ahead
 * @param window_size Number of sequences to prefetch (default: 5)
 * @return true on success, false on error
 */
bool llama_dataset_set_streaming_read_ahead(struct llama_dataset* dataset, bool enabled, size_t window_size) {
    if (!dataset || !dataset->streaming) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Dataset is not in streaming mode");
        return false;
    }

    // Get or create the optimization manager
    llama_dataset_stream_optimization_manager* manager = llama_dataset_streaming_get_optimization_manager(dataset);
    if (!manager) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Failed to initialize optimization manager");
        return false;
    }

    // Configure read-ahead
    manager->set_read_ahead_enabled(enabled);
    manager->set_read_ahead_window(window_size);

    LLAMA_LOG_INFO("Set streaming read-ahead to %s with window size %zu", enabled ? "enabled" : "disabled", window_size);
    return true;
}

/**
 * @brief Enable or disable adaptive cache sizing based on memory pressure.
 *
 * @param dataset Dataset to configure
 * @param enabled Whether to enable adaptive sizing
 * @return true on success, false on error
 */
bool llama_dataset_set_adaptive_cache_sizing(struct llama_dataset* dataset, bool enabled) {
    if (!dataset || !dataset->streaming) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Dataset is not in streaming mode");
        return false;
    }

    // Get the streaming cache
    llama_dataset_streaming_cache* cache = dataset->streaming_cache;
    if (!cache) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Streaming cache not initialized");
        return false;
    }

    // Update the optimization manager if it exists
    if (auto manager = llama_dataset_streaming_get_optimization_manager(dataset)) {
        manager->set_adaptive_cache_enabled(enabled);
    }

    LLAMA_LOG_INFO("Set adaptive cache sizing to %s", enabled ? "enabled" : "disabled");
    return true;
}

/**
 * @brief Get streaming cache statistics.
 *
 * @param dataset Dataset to query
 * @param hit_ratio Pointer to store hit ratio (0.0-1.0)
 * @param memory_usage_bytes Pointer to store current memory usage
 * @param entry_count Pointer to store number of entries in cache
 * @return true on success, false on error
 */
bool llama_dataset_get_streaming_stats(
    const struct llama_dataset* dataset,
    double* hit_ratio,
    size_t* memory_usage_bytes,
    size_t* entry_count) {

    if (!dataset || !dataset->streaming) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Dataset is not in streaming mode");
        return false;
    }

    // Get the streaming cache
    llama_dataset_streaming_cache* cache = dataset->streaming_cache;
    if (!cache) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Streaming cache not initialized");
        return false;
    }

    // Get the cache statistics
    llama_dataset_streaming_cache::CacheStats stats = cache->get_stats();

    // Set the output values
    if (hit_ratio) {
        *hit_ratio = stats.hit_ratio;
    }

    if (memory_usage_bytes) {
        *memory_usage_bytes = stats.current_memory;
    }

    if (entry_count) {
        *entry_count = stats.entry_count;
    }

    return true;
}
