#pragma once
#ifdef LLAMA_PARQUET
#include "../../core/llama-dataset.h"

#ifdef __cplusplus
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

// Forward declarations for Arrow types
namespace arrow {
    class Table;
    class Schema;
    class DataType;
}

// Forward declarations for llama types
struct llama_model;
struct llama_context;
#endif

/**
 * @brief Parquet dataset loader implementation.
 *
 * This header contains functions for loading Parquet datasets.
 */

#ifdef __cplusplus
/**
 * @brief Tokenization engine for Parquet processing.
 *
 * This class handles text-to-token conversion using llama tokenizer,
 * with caching support for efficiency.
 */
class llama_dataset_parquet_tokenizer {
public:
    /**
     * @brief Constructor with llama model integration.
     *
     * @param model Llama model for tokenization (must not be null)
     */
    explicit llama_dataset_parquet_tokenizer(struct llama_model * model);

    /**
     * @brief Destructor - cleans up resources.
     */
    ~llama_dataset_parquet_tokenizer();

    // Disable copy constructor and assignment operator
    llama_dataset_parquet_tokenizer(const llama_dataset_parquet_tokenizer &) = delete;
    llama_dataset_parquet_tokenizer & operator=(const llama_dataset_parquet_tokenizer &) = delete;

    /**
     * @brief Convert text to tokens using llama tokenizer.
     *
     * @param text Input text to tokenize
     * @return Vector of tokens, empty on error
     */
    std::vector<int32_t> tokenize_text(const std::string & text);

    /**
     * @brief Batch tokenization for efficiency.
     *
     * @param texts Vector of input texts to tokenize
     * @param results Output vector for tokenized results
     * @return true on success, false on error
     */
    bool tokenize_batch(const std::vector<std::string> & texts,
                       std::vector<std::vector<int32_t>> & results);

    /**
     * @brief Set tokenization cache size limit.
     *
     * @param max_size_mb Maximum cache size in megabytes
     */
    void set_cache_size(size_t max_size_mb);

    /**
     * @brief Clear tokenization cache.
     */
    void clear_cache();

    /**
     * @brief Get cache hit ratio for performance monitoring.
     *
     * @return Cache hit ratio (0.0 to 1.0)
     */
    double get_cache_hit_ratio() const;

    /**
     * @brief Get total number of tokens processed.
     *
     * @return Total token count
     */
    size_t get_total_tokens() const;

    /**
     * @brief Get number of unique texts processed.
     *
     * @return Unique text count
     */
    size_t get_unique_texts() const;

    /**
     * @brief Check if tokenizer is valid and ready to use.
     *
     * @return true if tokenizer is ready, false otherwise
     */
    bool is_valid() const;

private:
    struct llama_model * model_;                                        // Llama model reference
    struct llama_context * ctx_;                                        // Tokenizer context
    std::unordered_map<std::string, std::vector<int32_t>> cache_;      // Text-to-token cache
    size_t max_cache_size_;                                            // Max cache size in bytes
    size_t current_cache_size_;                                        // Current cache size in bytes
    size_t cache_hits_;                                                // Cache hit counter
    size_t cache_misses_;                                              // Cache miss counter
    size_t total_tokens_;                                              // Total tokens processed
    bool owns_context_;                                                // Whether we own the context

    /**
     * @brief Estimate memory usage of cached entry.
     *
     * @param text Input text
     * @param tokens Token vector
     * @return Estimated memory usage in bytes
     */
    size_t estimate_cache_entry_size(const std::string & text,
                                    const std::vector<int32_t> & tokens) const;

    /**
     * @brief Evict cache entries using LRU strategy.
     *
     * @param target_size Target cache size after eviction
     */
    void evict_cache_entries(size_t target_size);

    /**
     * @brief Internal tokenization implementation.
     *
     * @param text Input text
     * @param tokens Output token vector
     * @return true on success, false on error
     */
    bool tokenize_internal(const std::string & text, std::vector<int32_t> & tokens);
};
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure containing information about Parquet schema analysis.
 *
 * This structure holds the results of analyzing a Parquet file's schema
 * to determine which columns contain text data vs pre-tokenized data.
 */
struct parquet_schema_info {
    char ** text_columns;           // Array of text column names
    size_t n_text_columns;          // Number of text columns
    char ** token_columns;          // Array of token column names
    size_t n_token_columns;         // Number of token columns
    bool has_mixed_content;         // True if file has both text and token columns
    int primary_text_column_index;  // Index of primary text column (-1 if none)
    int primary_token_column_index; // Index of primary token column (-1 if none)
};

/**
 * @brief Free resources associated with parquet_schema_info.
 *
 * @param info Schema info structure to free
 */
void parquet_schema_info_free(struct parquet_schema_info * info);

/**
 * @brief Free Parquet format-specific data.
 *
 * This function cleans up all Parquet-specific resources including
 * schema analysis results and Arrow/Parquet objects.
 *
 * @param format_data Pointer to parquet_format_data to free
 */
void llama_dataset_free_parquet_format_data(void * format_data);


/**
 * @brief Load a dataset from a Parquet file.
 *
 * This function loads a dataset from a Parquet file, with an option to use streaming mode.
 * In streaming mode, data is loaded on demand, which can save memory for large datasets.
 *
 * @param params Common parameters including file path and streaming options
 * @return Pointer to the dataset, or NULL on error
 */
struct llama_dataset * llama_dataset_load_parquet(const common_params * params);

/**
 * @brief Validate Parquet file schema.
 *
 * This function validates that a Parquet file has the expected schema for dataset loading.
 *
 * @param path Path to the Parquet file
 * @param info Optional pointer to store schema analysis results (can be NULL)
 * @return true if schema is valid, false otherwise
 */
bool llama_dataset_validate_parquet_schema(const char * path, struct parquet_schema_info * info);

/**
 * @brief Analyze Parquet file schema for mixed content support.
 *
 * This function analyzes a Parquet file's schema to determine which columns
 * contain text data vs pre-tokenized data, enabling mixed content processing.
 *
 * @param path Path to the Parquet file
 * @param preferred_text_column Preferred text column name (can be NULL)
 * @param preferred_token_column Preferred token column name (can be NULL)
 * @param info Pointer to store schema analysis results
 * @return true if analysis successful, false otherwise
 */
bool analyze_parquet_schema(
    const char * path,
    const char * preferred_text_column,
    const char * preferred_token_column,
    struct parquet_schema_info * info
);

#ifdef __cplusplus
}
#endif
#endif
