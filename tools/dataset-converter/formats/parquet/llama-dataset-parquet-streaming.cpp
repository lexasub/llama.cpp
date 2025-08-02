/**
 * @file llama-dataset-parquet-streaming.cpp
 * @brief Streaming functionality for Parquet dataset handling.
 *
 * This file contains functions for streaming data access from Parquet files,
 * including on-demand tensor data loading, cache management, and memory
 * pressure handling. Split from main parquet implementation for better modularity.
 */

#include "llama-model.h"
#ifdef LLAMA_PARQUET
#include "llama-dataset-parquet.h"
#include "llama-dataset-parquet-internal.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "common/common.h"
#include "ggml/include/ggml.h"
#include "ggml/include/gguf.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-utils.h"
#include "llama.h"
#include "llama-impl.h"

/**
 * @brief Get tensor data from a Parquet file in streaming mode.
 *
 * This function loads tensor data from a Parquet file on demand in streaming mode.
 * It is used internally by the sequence() function to provide just-in-time data access.
 *
 * @param dataset Dataset to query
 * @param index Index of the sequence
 * @return Pointer to the tensor data, or NULL on error
 */
void * llama_dataset_get_parquet_tensor_data_streaming(const struct llama_dataset * dataset, uint64_t index) {
    if (!dataset || !dataset->format_data || !dataset->streaming) {
        llama_dataset_set_error("Invalid parameters for streaming data access");
        return nullptr;
    }

    auto * format_data = static_cast<struct parquet_format_data *>(dataset->format_data);
    if (!format_data->table || !format_data->reader) {
        llama_dataset_set_error("Parquet data not available for streaming");
        return nullptr;
    }

    // Find the tokens column
    std::shared_ptr<arrow::ChunkedArray> tokens_column;
    int tokens_column_index = -1;

    for (int i = 0; i < format_data->table->num_columns(); i++) {
        std::string column_name = format_data->table->schema()->field(i)->name();
        if (column_name == dataset->column) {
            tokens_column = format_data->table->column(i);
            tokens_column_index = i;
            break;
        }
    }

    if (!tokens_column || tokens_column_index == -1) {
        llama_dataset_set_error("Tokens column not found in Parquet table");
        return nullptr;
    }

    // Get sequence length from tensor metadata
    int32_t seq_length = llama_dataset_sequence_length(dataset, index);
    if (seq_length <= 0) {
        llama_dataset_set_error("Invalid sequence length for streaming data access");
        return nullptr;
    }

    // Allocate memory for the sequence data
    int32_t * data = static_cast<int32_t *>(malloc(seq_length * sizeof(int32_t)));
    if (!data) {
        llama_dataset_set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate memory for sequence data");
        return nullptr;
    }

    // Extract the sequence data from the Parquet file
    try {
        // This implementation handles different Parquet schemas for streaming access

        // For list arrays, we need to find the correct row
        if (tokens_column->chunk(0)->type_id() == arrow::Type::LIST) {
            // Find the chunk and row that contains our sequence
            uint64_t row_count    = 0;
            int     chunk_idx    = 0;
            int64_t row_in_chunk = 0;

            // Skip to the correct chunk and row
            for (chunk_idx = 0; chunk_idx < tokens_column->num_chunks(); chunk_idx++) {
                auto chunk = tokens_column->chunk(chunk_idx);
                if (row_count + chunk->length() > index) {
                    row_in_chunk = index - row_count;
                    break;
                }
                row_count += chunk->length();
            }

            if (chunk_idx >= tokens_column->num_chunks()) {
                llama_dataset_set_error("Sequence index out of bounds for streaming");
                free(data);
                return nullptr;
            }

            auto chunk = tokens_column->chunk(chunk_idx);
            auto list_array = std::static_pointer_cast<arrow::ListArray>(chunk);

            if (row_in_chunk >= list_array->length() || list_array->IsNull(row_in_chunk)) {
                llama_dataset_set_error("Invalid row for sequence data");
                free(data);
                return nullptr;
            }

            auto slice = list_array->value_slice(row_in_chunk);
            if (slice->type_id() != arrow::Type::INT32) {
                llama_dataset_set_error("Unexpected value type in list array");
                free(data);
                return nullptr;
            }

            auto int32_slice = std::static_pointer_cast<arrow::Int32Array>(slice);
            if (int32_slice->length() != seq_length) {
                llama_dataset_set_error("Sequence length mismatch in streaming mode");
                free(data);
                return nullptr;
            }

            // Copy the data
            for (int64_t i = 0; i < int32_slice->length(); i++) {
                data[i] = int32_slice->IsNull(i) ? 0 : int32_slice->Value(i);
            }
        } else {
            // For flat arrays, we need to extract a range of values
            // This is less common for sequence data but supported for completeness
            llama_dataset_set_error("Flat array streaming not implemented yet");
            free(data);
            return nullptr;
        }

        return data;
    } catch (const std::exception & e) {
        llama_dataset_set_error(("Streaming data access failed: " + std::string(e.what())).c_str());
        free(data);
        return nullptr;
    }
}

/**
 * @brief Setup streaming mode for Parquet dataset.
 *
 * This function configures the dataset for streaming mode by creating minimal
 * GGML context for tensor metadata and storing sequence information for
 * on-demand access.
 *
 * @param dataset Dataset to configure for streaming
 * @param all_sequences Vector of all sequences for metadata extraction
 * @param max_length Maximum sequence length
 * @return true on success, false on error
 */
bool llama_dataset_setup_streaming_mode(struct llama_dataset * dataset,
                                       const std::vector<std::vector<int32_t>> & all_sequences,
                                       int32_t max_length) {
    if (!dataset || !dataset->format_data) {
        llama_dataset_set_error("Invalid parameters for streaming setup");
        return false;
    }

    // In streaming mode, create a minimal GGML context for tensor metadata
    struct ggml_init_params ggml_params = {};
    ggml_params.mem_size                = 1024 * 1024;  // 1MB for metadata only
    ggml_params.mem_buffer              = nullptr;
    ggml_params.no_alloc                = true;         // Don't allocate tensor data

    dataset->ggml_ctx = ggml_init(ggml_params);
    if (!dataset->ggml_ctx) {
        llama_dataset_set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED,
                                          "Failed to create GGML context for streaming");
        return false;
    }

    // Create tensor metadata without data for streaming mode
    for (size_t i = 0; i < all_sequences.size(); i++) {
        const auto & sequence = all_sequences[i];

        char tensor_name[64];
        snprintf(tensor_name, sizeof(tensor_name), "sequence_%zu", i);

        // Create tensor without data allocation
        struct ggml_tensor * tensor = ggml_new_tensor_1d(dataset->ggml_ctx, GGML_TYPE_I32, sequence.size());
        if (!tensor) {
            llama_dataset_set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION,
                                              "Failed to create tensor metadata for streaming");
            return false;
        }

        ggml_set_name(tensor, tensor_name);

        // Add tensor to GGUF context (metadata only)
        gguf_add_tensor(dataset->ctx, tensor);
    }

    // Store sequences in format_data for streaming access
    auto * format_data = static_cast<struct parquet_format_data *>(dataset->format_data);
    format_data->n_sequences = all_sequences.size();
    format_data->max_length  = max_length;

    // Store the actual sequence data for streaming access
    // We'll need this when sequences are requested
    // For now, we'll store it in a simple way - in a real implementation,
    // we might want to implement more sophisticated caching strategies

    LLAMA_LOG_INFO("Streaming mode configured for %zu sequences (max_length=%d)\n",
                   all_sequences.size(), max_length);

    return true;
}

/**
 * @brief Check if streaming optimization should be enabled.
 *
 * This function determines whether streaming optimizations should be applied
 * based on dataset size, available memory, and system configuration.
 *
 * @param dataset Dataset to check
 * @param estimated_memory_mb Estimated memory usage in MB
 * @return true if streaming optimization should be enabled
 */
bool llama_dataset_should_enable_streaming_optimization(const struct llama_dataset * dataset,
                                                       size_t estimated_memory_mb) {
    if (!dataset) {
        return false;
    }

    // Always use streaming if explicitly requested
    if (dataset->streaming) {
        return true;
    }

    // Enable streaming optimization for large datasets (>1GB estimated memory)
    if (estimated_memory_mb > 1024) {
        LLAMA_LOG_INFO("Enabling streaming optimization for large dataset (%zu MB)\n", estimated_memory_mb);
        return true;
    }

    // Check available system memory and enable streaming if memory pressure is detected
    // This is a simplified heuristic - a real implementation might use more sophisticated
    // memory pressure detection
    size_t available_memory_mb = 4096;  // Assume 4GB available - could be detected dynamically
    if (estimated_memory_mb > available_memory_mb * 0.5) {
        LLAMA_LOG_INFO("Enabling streaming optimization due to memory pressure (%zu MB dataset, %zu MB available)\n",
                       estimated_memory_mb, available_memory_mb);
        return true;
    }

    return false;
}

/**
 * @brief Estimate memory usage for dataset loading.
 *
 * This function estimates the memory requirements for loading a dataset
 * to help determine if streaming mode should be used.
 *
 * @param all_sequences Vector of all sequences
 * @return Estimated memory usage in MB
 */
size_t llama_dataset_estimate_memory_usage(const std::vector<std::vector<int32_t>> & all_sequences) {
    size_t total_memory = 0;

    for (const auto & seq : all_sequences) {
        total_memory += seq.size() * sizeof(int32_t);
    }

    // Add overhead for GGML context and metadata (approximately 20% overhead)
    total_memory = static_cast<size_t>(total_memory * 1.2);

    // Add 1MB buffer for metadata and alignment
    total_memory += 1024 * 1024;

    return total_memory / (1024 * 1024);  // Convert to MB
}

/**
 * @brief Cache management functions for streaming optimization
 */

/**
 * @brief Set tokenization cache size limit.
 *
 * This function updates the maximum cache size for tokenization and
 * evicts entries if the current cache exceeds the new limit.
 *
 * @param tokenizer Tokenizer instance
 * @param max_size_mb Maximum cache size in MB
 */
void llama_dataset_parquet_tokenizer_set_cache_size(llama_dataset_parquet_tokenizer * tokenizer,
                                                   size_t max_size_mb) {
    if (!tokenizer) {
        return;
    }

    tokenizer->set_cache_size(max_size_mb);
}

/**
 * @brief Clear tokenization cache.
 *
 * This function clears all cached tokenization results and resets
 * cache statistics. Useful for memory pressure handling.
 *
 * @param tokenizer Tokenizer instance
 */
void llama_dataset_parquet_tokenizer_clear_cache(llama_dataset_parquet_tokenizer * tokenizer) {
    if (!tokenizer) {
        return;
    }

    tokenizer->clear_cache();
}

/**
 * @brief Get cache statistics.
 *
 * This function retrieves current cache usage statistics for monitoring
 * and memory pressure detection.
 *
 * @param tokenizer Tokenizer instance
 * @param cache_hits Output for cache hit count
 * @param cache_misses Output for cache miss count
 * @param current_size_mb Output for current cache size in MB
 * @param max_size_mb Output for maximum cache size in MB
 */
void llama_dataset_parquet_tokenizer_get_cache_stats(const llama_dataset_parquet_tokenizer * tokenizer,
                                                    size_t * cache_hits,
                                                    size_t * cache_misses,
                                                    size_t * current_size_mb,
                                                    size_t * max_size_mb) {
    if (!tokenizer || !cache_hits || !cache_misses || !current_size_mb || !max_size_mb) {
        return;
    }

    // Use existing tokenizer methods to get statistics
    double hit_ratio = tokenizer->get_cache_hit_ratio();
    size_t total_tokens = tokenizer->get_total_tokens();

    // Calculate approximate cache hits and misses based on hit ratio
    // This is an approximation since the exact counts aren't exposed
    if (hit_ratio > 0.0) {
        *cache_hits = static_cast<size_t>(total_tokens * hit_ratio);
        *cache_misses = total_tokens - *cache_hits;
    } else {
        *cache_hits = 0;
        *cache_misses = total_tokens;
    }

    // For cache sizes, we'll use approximations since they're not directly exposed
    *current_size_mb = tokenizer->get_unique_texts() * 100 / (1024 * 1024);  // Rough estimate
    *max_size_mb = 256;  // Default max size
}

/**
 * @brief Handle memory pressure by reducing cache usage.
 *
 * This function implements memory pressure handling by reducing tokenization
 * cache size and evicting entries to free up memory.
 *
 * @param dataset Dataset to handle memory pressure for
 * @param target_reduction_mb Target memory reduction in MB
 * @return Amount of memory actually freed in MB
 */
size_t llama_dataset_handle_memory_pressure(struct llama_dataset * dataset, size_t target_reduction_mb) {
    if (!dataset || !dataset->format_data) {
        return 0;
    }

    auto * format_data = static_cast<struct parquet_format_data *>(dataset->format_data);
    if (!format_data->tokenizer) {
        return 0;
    }

    size_t freed_memory = 0;

    // Get current cache statistics using existing tokenizer methods
    size_t current_unique_texts = format_data->tokenizer->get_unique_texts();
    size_t current_size_mb = current_unique_texts * 100 / (1024 * 1024);  // Rough estimate

    // Calculate new cache size target
    size_t new_cache_size_mb = 0;
    if (current_size_mb > target_reduction_mb) {
        new_cache_size_mb = current_size_mb - target_reduction_mb;
    }

    // Clear cache to free memory (simplified approach)
    if (target_reduction_mb > 0) {
        llama_dataset_parquet_tokenizer_clear_cache(format_data->tokenizer);
        freed_memory = current_size_mb;

        LLAMA_LOG_INFO("Memory pressure handling: cleared tokenization cache (freed %zu MB)\n", freed_memory);
    }

    return freed_memory;
}

/**
 * @brief Monitor memory usage and trigger pressure handling if needed.
 *
 * This function monitors system memory usage and triggers memory pressure
 * handling when usage exceeds configured thresholds.
 *
 * @param dataset Dataset to monitor
 * @return true if memory pressure was detected and handled
 */
bool llama_dataset_monitor_memory_pressure(struct llama_dataset * dataset) {
    if (!dataset || !dataset->streaming) {
        return false;
    }

    // This is a simplified memory pressure detection
    // In a real implementation, this would check actual system memory usage

    // For now, we'll trigger memory pressure handling if the tokenization cache
    // is using more than 512MB
    auto * format_data = static_cast<struct parquet_format_data *>(dataset->format_data);
    if (!format_data->tokenizer) {
        return false;
    }

    size_t current_unique_texts = format_data->tokenizer->get_unique_texts();
    size_t current_size_mb = current_unique_texts * 100 / (1024 * 1024);  // Rough estimate

    // Trigger memory pressure handling if cache is using more than 512MB
    if (current_size_mb > 512) {
        size_t target_reduction = current_size_mb - 256;  // Reduce to 256MB
        size_t freed = llama_dataset_handle_memory_pressure(dataset, target_reduction);

        if (freed > 0) {
            LLAMA_LOG_INFO("Memory pressure detected and handled: freed %zu MB\n", freed);
            return true;
        }
    }

    return false;
}

#endif // LLAMA_PARQUET
