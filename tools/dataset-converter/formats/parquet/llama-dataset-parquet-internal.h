#pragma once

/**
 * @file llama-dataset-parquet-internal.h
 * @brief Internal structures and declarations for Parquet dataset handling.
 *
 * This header contains internal data structures and function declarations
 * shared between different Parquet module files. It should only be included
 * by implementation files (.cpp) within the Parquet format handler.
 */

#ifdef LLAMA_PARQUET

#include <stdint.h>
#include <stdbool.h>
#include "llama-dataset-parquet.h"
#include "core/llama-dataset-internal.h"

#ifdef __cplusplus
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

// Forward declarations for Arrow types
namespace arrow {
    class Table;
    class DataType;
    class Array;
}

namespace parquet {
    namespace arrow {
        class FileReader;
    }
}

// Forward declaration for tokenizer
class llama_dataset_parquet_tokenizer;

extern "C" {
#endif

/**
 * @brief Parquet-specific format data structure.
 *
 * This structure holds Parquet-specific state for both regular and streaming modes.
 * It contains Arrow/Parquet objects, schema analysis results, and tokenization support.
 */
struct parquet_format_data {
#ifdef __cplusplus
    std::shared_ptr<arrow::Table>               table;
    std::shared_ptr<parquet::arrow::FileReader> reader;
    std::string                                 file_path;
#else
    void *                                      table;      // Opaque pointer for C compatibility
    void *                                      reader;     // Opaque pointer for C compatibility
    char *                                      file_path;  // C-style string
#endif
    uint64_t                                    n_sequences;
    int32_t                                     max_length;

    // Schema analysis results
    struct parquet_schema_info                  schema_info;
    bool                                        schema_analyzed;

    // Tokenization support
#ifdef __cplusplus
    llama_dataset_parquet_tokenizer *           tokenizer;
    std::vector<std::string>                    text_columns;
    std::vector<std::string>                    token_columns;
    std::unordered_map<uint64_t, std::vector<int32_t>> tokenized_cache;
#else
    void *                                      tokenizer;      // Opaque pointer for C compatibility
    void *                                      text_columns;   // Opaque pointer for C compatibility
    void *                                      token_columns;  // Opaque pointer for C compatibility
    void *                                      tokenized_cache; // Opaque pointer for C compatibility
#endif
    bool                                        mixed_content;
};

//
// Forward declarations for internal functions
//

/**
 * @brief Schema analysis functions (implemented in llama-dataset-parquet-schema.cpp)
 */

/**
 * @brief Check if a column contains text data based on its type.
 *
 * @param field_type Arrow data type to check
 * @return true if the column contains text data, false otherwise
 */
#ifdef __cplusplus
bool is_text_column_type(const std::shared_ptr<arrow::DataType> & field_type);
#endif

/**
 * @brief Check if a column contains token data based on its type.
 *
 * @param field_type Arrow data type to check
 * @return true if the column contains token data, false otherwise
 */
#ifdef __cplusplus
bool is_token_column_type(const std::shared_ptr<arrow::DataType> & field_type);
#endif

/**
 * @brief Analyze Parquet table schema for mixed content support (C++ version).
 *
 * @param table Arrow table to analyze
 * @param preferred_text_column Preferred text column name (empty for auto-detection)
 * @param preferred_token_column Preferred token column name (empty for auto-detection)
 * @param info Output structure for schema information
 * @return true on success, false on error
 */
#ifdef __cplusplus
bool analyze_parquet_table_schema(
    const std::shared_ptr<arrow::Table> & table,
    const std::string & preferred_text_column,
    const std::string & preferred_token_column,
    struct parquet_schema_info * info
);
#endif

/**
 * @brief Data conversion functions (implemented in llama-dataset-parquet-conversion.cpp)
 */

/**
 * @brief Convert Arrow array to token vector.
 *
 * @param array Arrow array containing token data
 * @param tokens Output vector for tokens
 * @return true on success, false on error
 */
#ifdef __cplusplus
bool llama_dataset_arrow_array_to_tokens(const std::shared_ptr<arrow::Array> & array,
                                        std::vector<int32_t> & tokens);
#endif

/**
 * @brief Create GGUF dataset from Parquet table.
 *
 * @param table Arrow table containing the data
 * @param dataset Output dataset structure
 * @return true on success, false on error
 */
#ifdef __cplusplus
bool llama_dataset_create_gguf_from_parquet(const std::shared_ptr<arrow::Table> & table,
                                           struct llama_dataset * dataset);
#endif

/**
 * @brief Streaming functions (implemented in llama-dataset-parquet-streaming.cpp)
 */

/**
 * @brief Get tensor data for streaming mode.
 *
 * @param dataset Dataset to query
 * @param index Index of the sequence
 * @return Pointer to tensor data, or NULL on error
 */
void * llama_dataset_get_parquet_tensor_data_streaming(const struct llama_dataset * dataset,
                                                     uint64_t index);

/**
 * @brief Setup streaming mode for Parquet dataset.
 *
 * @param dataset Dataset to configure for streaming
 * @param all_sequences Vector of all sequences for metadata extraction
 * @param max_length Maximum sequence length
 * @return true on success, false on error
 */
#ifdef __cplusplus
bool llama_dataset_setup_streaming_mode(struct llama_dataset * dataset,
                                       const std::vector<std::vector<int32_t>> & all_sequences,
                                       int32_t max_length);
#endif

/**
 * @brief Check if streaming optimization should be enabled.
 *
 * @param dataset Dataset to check
 * @param estimated_memory_mb Estimated memory usage in MB
 * @return true if streaming optimization should be enabled
 */
bool llama_dataset_should_enable_streaming_optimization(const struct llama_dataset * dataset,
                                                       size_t estimated_memory_mb);

/**
 * @brief Estimate memory usage for dataset loading.
 *
 * @param all_sequences Vector of all sequences
 * @return Estimated memory usage in MB
 */
#ifdef __cplusplus
size_t llama_dataset_estimate_memory_usage(const std::vector<std::vector<int32_t>> & all_sequences);
#endif

/**
 * @brief Cache management and memory pressure handling functions
 */

/**
 * @brief Set tokenization cache size limit.
 *
 * @param tokenizer Tokenizer instance
 * @param max_size_mb Maximum cache size in MB
 */
#ifdef __cplusplus
void llama_dataset_parquet_tokenizer_set_cache_size(llama_dataset_parquet_tokenizer * tokenizer,
                                                   size_t max_size_mb);
#endif

/**
 * @brief Clear tokenization cache.
 *
 * @param tokenizer Tokenizer instance
 */
#ifdef __cplusplus
void llama_dataset_parquet_tokenizer_clear_cache(llama_dataset_parquet_tokenizer * tokenizer);
#endif

/**
 * @brief Get cache statistics.
 *
 * @param tokenizer Tokenizer instance
 * @param cache_hits Output for cache hit count
 * @param cache_misses Output for cache miss count
 * @param current_size_mb Output for current cache size in MB
 * @param max_size_mb Output for maximum cache size in MB
 */
#ifdef __cplusplus
void llama_dataset_parquet_tokenizer_get_cache_stats(const llama_dataset_parquet_tokenizer * tokenizer,
                                                    size_t * cache_hits,
                                                    size_t * cache_misses,
                                                    size_t * current_size_mb,
                                                    size_t * max_size_mb);
#endif

/**
 * @brief Handle memory pressure by reducing cache usage.
 *
 * @param dataset Dataset to handle memory pressure for
 * @param target_reduction_mb Target memory reduction in MB
 * @return Amount of memory actually freed in MB
 */
size_t llama_dataset_handle_memory_pressure(struct llama_dataset * dataset, size_t target_reduction_mb);

/**
 * @brief Monitor memory usage and trigger pressure handling if needed.
 *
 * @param dataset Dataset to monitor
 * @return true if memory pressure was detected and handled
 */
bool llama_dataset_monitor_memory_pressure(struct llama_dataset * dataset);

/**
 * @brief Core functions (implemented in llama-dataset-parquet-core.cpp)
 */

/**
 * @brief Internal Parquet loading function.
 *
 * @param path Path to the Parquet file
 * @param streaming Whether to use streaming mode
 * @param preferred_text_column Preferred text column name (can be NULL)
 * @param preferred_token_column Preferred token column name (can be NULL)
 * @param model Llama model for tokenization (can be NULL)
 * @return Pointer to the dataset, or NULL on error
 */
struct llama_dataset * llama_dataset_load_parquet_internal(const common_params * params);

/**
 * @brief Validate Parquet schema for dataset compatibility.
 *
 * @param path Path to the Parquet file
 * @param info Output structure for schema information (can be NULL)
 * @return true if schema is valid, false otherwise
 */
bool llama_dataset_validate_parquet_schema(const char * path, struct parquet_schema_info * info);

/**
 * @brief Get metadata from Parquet file.
 *
 * @param path Path to the Parquet file
 * @param key Metadata key to retrieve
 * @return Metadata value as string, or NULL if not found
 */
bool llama_dataset_get_parquet_metadata(const char * path, uint64_t * n_sequences, int32_t * max_length);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_PARQUET
