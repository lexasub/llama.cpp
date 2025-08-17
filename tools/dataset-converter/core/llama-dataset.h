#pragma once

/**
 * @file llama-dataset.h
 * @brief Core dataset interface for the llama.cpp dataset converter framework.
 *
 * This header provides the primary C interface for working with training datasets in multiple
 * formats (GGUF, text, Parquet) with comprehensive streaming, validation, and optimization
 * capabilities. It serves as the central API for the modular dataset converter architecture.
 *
 * ## Architecture Overview
 *
 * The dataset converter is built with a modular architecture consisting of:
 * - **Core Module** (this file): Primary dataset interface and data structures
 * - **Format Modules**: Specialized loaders for GGUF, text, and Parquet formats
 * - **Streaming Module**: Advanced streaming capabilities with caching and optimization
 * - **Validation Module**: Comprehensive data integrity and format validation
 * - **Platform Module**: Cross-platform compatibility and system integration
 * - **Tools Module**: Command-line utilities and analysis tools
 *
 * ## Key Features
 *
 * - **Multi-format Support**: Native support for GGUF, text, and Parquet datasets
 * - **Streaming Architecture**: Memory-efficient streaming with configurable caching
 * - **Adaptive Optimization**: Dynamic cache sizing and read-ahead buffering
 * - **Comprehensive Validation**: Format-specific and cross-format data validation
 * - **Metadata Management**: Rich metadata support with standardized key definitions
 * - **Error Handling**: Robust error reporting with detailed diagnostic information
 * - **Performance Monitoring**: Built-in statistics and performance metrics
 *
 * ## Usage Patterns
 *
 * ### Basic Dataset Loading
 * ```c
 * // Load a GGUF dataset
 * struct llama_dataset* dataset = llama_dataset_from_gguf(params);
 *
 * // Access sequences
 * uint64_t count = llama_dataset_n_sequences(dataset);
 * const int32_t* tokens = llama_dataset_sequence(dataset, 0);
 *
 * // Cleanup
 * llama_dataset_free(dataset);
 * ```
 *
 * ### Streaming Configuration
 * ```c
 * // Configure streaming optimizations
 * llama_dataset_set_streaming_cache_size(dataset, 1024 * 1024 * 100); // 100MB cache
 * llama_dataset_set_streaming_read_ahead(dataset, true, 10);           // Prefetch 10 sequences
 * llama_dataset_set_adaptive_cache_sizing(dataset, true);             // Enable adaptive sizing
 * ```
 *
 * ### Performance Monitoring
 * ```c
 * // Get streaming statistics
 * double hit_ratio;
 * size_t memory_usage, entry_count;
 * llama_dataset_get_streaming_stats(dataset, &hit_ratio, &memory_usage, &entry_count);
 * ```
 *
 * ## Integration with Other Modules
 *
 * This core interface integrates seamlessly with:
 * - **streaming: Provides streaming cache management and optimization
 * - **validation: Offers comprehensive dataset validation capabilities
 * - **formats: Implements format-specific loading and conversion logic
 * - **platform: Ensures cross-platform compatibility and system integration
 * - **tools: Provides command-line utilities and analysis tools
 *
 * ## Thread Safety
 *
 * The dataset interface is designed to be thread-safe for read operations when properly
 * synchronized. Write operations and configuration changes should be performed from a
 * single thread or with appropriate external synchronization.
 *
 * ## Memory Management
 *
 * The interface follows RAII principles where applicable. All resources are automatically
 * managed through the dataset lifecycle, with explicit cleanup via llama_dataset_free().
 * Streaming mode provides additional memory efficiency for large datasets.
 *
 * @see streaming/streaming-cache.h for streaming implementation details
 * @see validation/llama-dataset-validation.h for validation capabilities
 * @see formats/ directory for format-specific implementations
 * @see tools/ directory for command-line utilities and analysis tools
 *
 * @version 1.0
 * @since 2024
 */

#include <stdint.h>
#include "ggml.h"
#include "llama.h"

// Forward declarations for cross-module compatibility
struct common_params;
struct llama_model;
struct llama_context;
struct gguf_context;
struct ggml_context;
struct ggml_tensor;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Dataset structure for storing and accessing training data.
 *
 * This structure is opaque to the user and should only be accessed through the provided functions.
 * The internal implementation varies by format and includes:
 *
 * - **GGUF datasets**: GGUF context, GGML context, and cached tensor pointers
 * - **Text datasets**: Tokenized sequences with llama model integration
 * - **Parquet datasets**: Apache Arrow integration with schema management
 *
 * All formats support:
 * - Streaming capabilities with configurable caching
 * - Metadata management and access
 * - Performance monitoring and statistics
 * - Validation and integrity checking
 * - Cross-platform compatibility
 *
 * The structure automatically manages memory allocation, streaming cache,
 * and format-specific resources throughout its lifecycle.
 */
struct llama_dataset;

/**
 * @brief Standard metadata key definitions for dataset properties.
 *
 * These standardized keys ensure consistent metadata access across all supported
 * formats and enable interoperability between different dataset sources.
 *
 * ## Core Metadata Keys
 * The following keys are supported across all dataset formats:
 *
 * ## Extended Metadata (Future)
 * Additional metadata keys planned for future implementation:
 * - training.dataset.source: string (optional) - URL or description of the data source
 * - training.tokenizer.gguf.vocab: array[string] - Tokenizer dictionary
 * - training.tokenizer.gguf.merges: array[string] - Tokenizer merges (for BPE)
 * - training.tokenizer.gguf.pre: string (optional) - Pre-tokenization architecture
 */
#define TRAINING_FORMAT_VERSION    "training.format.version"      // int16 (e.g. 1000) - Specification version, in case of future changes.
#define TRAINING_FORMAT_SOURCE     "training.format.source"       // Source format (gguf, text, parquet)
#define TRAINING_DATASET_NAME      "training.dataset.name"        // string (optional) - Dataset name (e.g. "OpenWebText-ru").
#define TRAINING_DATASET_DESCRIPTION "training.dataset.description" // string (optional) - Dataset description (e.g. "OpenWebText-ru").
#define TRAINING_SEQUENCE_COUNT    "training.sequence.count"      // Number of sequences in the dataset
#define TRAINING_MAX_LENGTH        "dataset.max_length"           // Maximum sequence length (verify key name)
#define TRAINING_TOKENIZER         "training.tokenizer.gguf.model"// Tokenizer model name (llama, gpt2, etc.).
#define TRAINING_CREATION_TIME     "training.file.creation_date"  // string (ISO 8601) - File creation date.

/**
 * @brief Dataset type enumeration for format identification.
 *
 * Each format has specific capabilities and requirements:
 * - GGUF: Native format with full streaming and metadata support
 * - Parquet: Requires Apache Arrow, supports complex schemas and streaming
 * - Text: Requires tokenization model, supports streaming with caching
 */
enum dataset_type {
    DATASET_GGUF,     ///< GGUF format (native, full feature support)
    DATASET_PARQUET,  ///< Parquet format (requires Arrow/Parquet support)
    DATASET_TEXT      ///< Text format (requires tokenization model)
};

/**
 * @brief Error codes for dataset operations.
 *
 * Comprehensive error reporting enables detailed diagnostics and proper
 * error handling across all modules and formats.
 */
enum dataset_error {
    DATASET_SUCCESS = 0,                ///< No error occurred
    DATASET_ERROR_FILE_NOT_FOUND,       ///< File not found or inaccessible
    DATASET_ERROR_INVALID_FORMAT,       ///< Invalid or corrupted file format
    DATASET_ERROR_MEMORY_ALLOCATION,    ///< Memory allocation failed
    DATASET_ERROR_TOKENIZATION_FAILED,  ///< Text tokenization failed
    DATASET_ERROR_STREAMING_NOT_SUPPORTED, ///< Streaming not supported for this format/file
    DATASET_ERROR_INVALID_PARAMETER,    ///< Invalid parameter passed to function
    DATASET_ERROR_CONTEXT_CREATION_FAILED, ///< Context creation failed (GGML/GGUF)
    DATASET_ERROR_IO_ERROR,             ///< General I/O error during file operations
    DATASET_ERROR_REGISTRY_FULL,        ///< Registry capacity exceeded, cannot register more loaders
    DATASET_ERROR_REGISTRY_DUPLICATE,   ///< Duplicate format loader registration attempted
    DATASET_ERROR_REGISTRY_NOT_FOUND,   ///< Format loader not found in registry
    DATASET_ERROR_REGISTRY_VALIDATION_FAILED, ///< Format loader validation failed during registration
    DATASET_ERROR_UNKNOWN               ///< Unknown or unspecified error
};

//
// Core Dataset Loading Interface
//
// These functions provide the primary entry points for loading datasets from
// different formats. Each function is implemented by the corresponding format
// module and integrates with the streaming and validation subsystems.
//

/**
 * @brief Load a dataset from a GGUF file.
 *
 * Loads a dataset from a GGUF file with full support for streaming, metadata,
 * and validation. GGUF is the native format with optimal performance and
 * feature support.
 *
 * @param params Common parameters including file path and streaming options
 * @return Pointer to the dataset, or NULL on error
 * @see formats/gguf/llama-dataset-gguf.h for GGUF-specific implementation
 */
struct llama_dataset * llama_dataset_from_gguf(const common_params * params);

/**
 * @brief Load a dataset from a text file and tokenize it.
 *
 * Loads and tokenizes a text file using the specified llama model. Supports
 * streaming with intelligent caching of tokenized sequences for memory efficiency.
 * The tokenization process is optimized for training data preparation.
 *
 * @param params Common parameters including file path and processing options
 * @param model Model to use for tokenization (must be compatible)
 * @return Pointer to the dataset, or NULL on error
 * @see formats/text/llama-dataset-text.h for text-specific implementation
 */
struct llama_dataset * llama_dataset_from_txt(const common_params * params, struct llama_model * model);

/**
 * @brief Load a dataset from a Parquet file.
 *
 * Loads a dataset from a Parquet file using Apache Arrow integration. Supports
 * complex schemas, streaming access, and automatic schema analysis. Requires
 * LLAMA_PARQUET to be defined at compile time.
 *
 * @param params Common parameters including file path and schema options
 * @return Pointer to the dataset, or NULL on error
 * @see formats/parquet/llama-dataset-parquet.h for Parquet-specific implementation
 */
#ifdef LLAMA_PARQUET
struct llama_dataset * llama_dataset_from_parquet(const common_params * params);

/**
 * @brief Load a dataset from a Parquet file with tokenization support.
 *
 * This function extends the basic Parquet loading functionality to support real-time
 * tokenization of raw text data stored in Parquet files. It can handle both pre-tokenized
 * data and raw text, automatically detecting the content type and applying appropriate
 * processing strategies.
 *
 * Key features:
 * - Automatic schema analysis to detect text vs token columns
 * - Real-time tokenization using the provided llama model
 * - Streaming support with intelligent tokenization caching
 * - Mixed content support (both text and pre-tokenized columns)
 * - Memory-efficient processing with configurable cache limits
 *
 * @param params Common parameters including file path, streaming options, and tokenization configuration
 * @param model Llama model to use for tokenization (required for text processing, can be NULL for pre-tokenized data)
 * @return Pointer to the dataset, or NULL on error
 * @see llama_dataset_set_tokenization_options() for configuration after loading
 * @see llama_dataset_get_tokenization_stats() for performance monitoring
 */
struct llama_dataset * llama_dataset_from_parquet_with_tokenization(
    const common_params * params,
    struct llama_model * model
);
#endif
/**
 * @brief Save a dataset to a GGUF file.
 *
 * Converts and saves any dataset format to GGUF format, preserving metadata
 * and ensuring optimal structure for training. The conversion process handles
 * format-specific optimizations and validation.
 *
 * @param dataset Dataset to save (any supported format)
 * @param path Path to the output GGUF file
 * @see tools/convert-to-gguf.cpp for command-line conversion utility
 */
void llama_dataset_to_gguf(struct llama_dataset * dataset, const char * path);

/**
 * @brief Get the number of sequences in the dataset.
 *
 * @param dataset Dataset to query
 * @return Number of sequences, or 0 if dataset is NULL
 */
uint64_t llama_dataset_n_sequences(const struct llama_dataset * dataset);

/**
 * @brief Get a pointer to the tokens in a sequence.
 *
 * @param dataset Dataset to query
 * @param index Index of the sequence
 * @return Pointer to the tokens, or NULL if dataset is NULL or index is out of bounds
 */
const int32_t * llama_dataset_sequence(const struct llama_dataset * dataset, uint64_t index);

/**
 * @brief Get a pointer to the tensor for a sequence.
 *
 * This is useful for advanced operations that need direct access to the tensor.
 *
 * @param dataset Dataset to query
 * @param index Index of the sequence
 * @return Pointer to the tensor, or NULL if dataset is NULL or index is out of bounds
 */
struct ggml_tensor * llama_dataset_sequence_tensor(const struct llama_dataset * dataset, uint64_t index);

/**
 * @brief Free resources associated with a dataset.
 *
 * @param dataset Dataset to free
 */
void llama_dataset_free(struct llama_dataset * dataset);

//
// Metadata Access Interface
//
// These functions provide standardized access to dataset metadata across all
// supported formats. Metadata is automatically extracted during loading and
// can include format-specific and user-defined properties.
//

/**
 * @brief Get a string metadata value from the dataset.
 *
 * @param dataset Dataset to query
 * @param key Metadata key
 * @return String value, or NULL if not found
 */
const char * llama_dataset_get_metadata_str(const struct llama_dataset * dataset, const char * key);

/**
 * @brief Get an integer metadata value from the dataset.
 *
 * @param dataset Dataset to query
 * @param key Metadata key
 * @param default_value Default value to return if key is not found
 * @return Integer value, or default_value if not found
 */
int64_t llama_dataset_get_metadata_int(const struct llama_dataset * dataset, const char * key, int64_t default_value);

/**
 * @brief Get a float metadata value from the dataset.
 *
 * @param dataset Dataset to query
 * @param key Metadata key
 * @param default_value Default value to return if key is not found
 * @return Float value, or default_value if not found
 */
float llama_dataset_get_metadata_float(const struct llama_dataset * dataset, const char * key, float default_value);

//
// Error Handling Interface
//
// Comprehensive error handling with detailed diagnostic information.
// Error state is maintained globally and can be queried after any operation.
// Thread-local storage ensures thread safety in multi-threaded environments.
//

/**
 * @brief Get the error message from the last operation.
 *
 * @return Error message, or NULL if no error
 */
const char * llama_dataset_get_error(void);

/**
 * @brief Check if an error occurred in the last operation.
 *
 * @return true if an error occurred, false otherwise
 */
bool llama_dataset_has_error(void);

/**
 * @brief Get the error message from the last operation.
 *
 * @return Error message, or empty string if no error
 */
const char * llama_dataset_get_error_message(void);

/**
 * @brief Get the error code from the last operation.
 *
 * @return Error code, or DATASET_SUCCESS if no error
 */
enum dataset_error llama_dataset_get_error_code(void);

/**
 * @brief Get a string representation of an error code.
 *
 * @param code Error code
 * @return String representation of the error code
 */
const char * llama_dataset_error_code_to_string(enum dataset_error code);

/**
 * @brief Clear the error state.
 */
void llama_dataset_clear_error(void);

/**
 * @brief Set an error message.
 *
 * @param message Error message to set
 */
void llama_dataset_set_error(const char * message);

/**
 * @brief Set an error message with a specific error code.
 *
 * @param code Error code
 * @param message Error message to set
 */
void llama_dataset_set_error_with_code(enum dataset_error code, const char * message);

//
// Tokenization Configuration and Monitoring Interface
//
// These functions provide comprehensive control over tokenization behavior and
// performance monitoring for datasets that support text-to-token conversion.
// They integrate with the streaming subsystem to provide optimal performance
// and memory usage for tokenization operations.
//

/**
 * @brief Configure tokenization options for a dataset.
 *
 * This function allows fine-tuning of tokenization behavior after dataset creation.
 * It provides control over caching strategies, memory usage, and column selection
 * for optimal performance based on specific use cases and system constraints.
 *
 * Configuration options include:
 * - Tokenization cache enable/disable and size limits
 * - Text and token column name specification
 * - Memory pressure handling strategies
 * - Batch processing parameters
 *
 * @param dataset Dataset to configure (must support tokenization)
 * @param enable_caching Whether to enable tokenization result caching
 * @param max_cache_size_mb Maximum cache size in megabytes (0 = unlimited, subject to memory pressure)
 * @param text_column_name Name of the text column to tokenize (NULL = use default from params)
 * @return true on success, false on error (check llama_dataset_get_error() for details)
 * @see llama_dataset_get_tokenization_stats() for monitoring cache performance
 */
bool llama_dataset_set_tokenization_options(
    struct llama_dataset * dataset,
    bool enable_caching,
    size_t max_cache_size_mb,
    const char * text_column_name
);

/**
 * @brief Get comprehensive tokenization statistics and performance metrics.
 *
 * This function provides detailed statistics about tokenization performance,
 * cache efficiency, and memory usage. The metrics are useful for performance
 * tuning, monitoring production systems, and debugging tokenization issues.
 *
 * Statistics include:
 * - Total number of tokens processed across all sequences
 * - Number of unique text strings processed (cache entries)
 * - Cache hit ratio for performance assessment
 * - Current memory usage by tokenization cache
 * - Processing time statistics (if available)
 *
 * @param dataset Dataset to query (must support tokenization)
 * @param total_tokens Pointer to store total number of tokens processed (can be NULL)
 * @param unique_texts Pointer to store number of unique text strings processed (can be NULL)
 * @param cache_hit_ratio Pointer to store cache hit ratio 0.0-1.0, higher is better (can be NULL)
 * @param cache_memory_usage_bytes Pointer to store current cache memory usage in bytes (can be NULL)
 * @return true on success, false on error or if tokenization is not supported
 * @see llama_dataset_set_tokenization_options() for cache configuration
 */
bool llama_dataset_get_tokenization_stats(
    const struct llama_dataset * dataset,
    size_t * total_tokens,
    size_t * unique_texts,
    double * cache_hit_ratio,
    size_t * cache_memory_usage_bytes
);


//
// Advanced Dataset Operations
//
// These functions provide advanced capabilities including streaming configuration,
// performance monitoring, and format-specific optimizations. They integrate with
// the streaming and validation subsystems to provide comprehensive dataset management.
//

// llama_dataset_load_gguf is implemented in formats/gguf/llama-dataset-gguf.h

/**
 * @brief Get the length of a sequence in the dataset (legacy function).
 *
 * @param dataset Dataset to query
 * @param index Index of the sequence
 * @return Length of the sequence, or 0 if dataset is NULL or index is out of bounds
 */
int32_t llama_dataset_sequence_length(const struct llama_dataset * dataset, uint64_t index);

/**
 * @brief Check if streaming is supported for a dataset type and file.
 *
 * @param type Dataset type
 * @param path Path to the file (optional, can be NULL)
 * @return true if streaming is supported, false otherwise
 */
bool llama_dataset_supports_streaming(enum dataset_type type, const char * path);

/**
 * @brief Check if streaming is enabled for a dataset.
 *
 * @param dataset Dataset to query
 * @return true if streaming is enabled, false otherwise
 */
bool llama_dataset_is_streaming_enabled(const struct llama_dataset * dataset);

/**
 * @brief Get the type of a dataset.
 *
 * @param dataset Dataset to query
 * @return Dataset type, or DATASET_GGUF if dataset is NULL
 */
enum dataset_type llama_dataset_get_type(const struct llama_dataset * dataset);

//
// Streaming Optimization Interface
//
// Advanced streaming configuration and monitoring capabilities. These functions
// integrate with the streaming subsystem to provide fine-grained control over
// caching behavior, memory usage, and performance optimization strategies.
//
// @see streaming/ directory for detailed streaming implementation
//

/**
 * @brief Configure streaming cache size for a dataset.
 *
 * Sets the maximum memory usage for the streaming cache. The cache uses LRU
 * eviction and can be dynamically resized. Larger caches improve performance
 * for random access patterns but consume more memory.
 *
 * @param dataset Dataset to configure
 * @param cache_size_bytes Maximum cache size in bytes (0 disables caching)
 * @return true on success, false on error
 * @see streaming/streaming-cache.h for cache implementation details
 */
bool llama_dataset_set_streaming_cache_size(struct llama_dataset * dataset, size_t cache_size_bytes);

/**
 * @brief Enable or disable read-ahead buffering for streaming.
 *
 * Configures predictive loading to prefetch sequences based on access patterns.
 * Read-ahead significantly improves performance for sequential access and can
 * adapt to detected access patterns for optimal prefetching.
 *
 * @param dataset Dataset to configure
 * @param enabled Whether to enable read-ahead buffering
 * @param window_size Number of sequences to prefetch (default: 5, max: 100)
 * @return true on success, false on error
 * @see streaming/streaming-read-ahead.h for read-ahead implementation
 */
bool llama_dataset_set_streaming_read_ahead(struct llama_dataset * dataset, bool enabled, size_t window_size);

/**
 * @brief Enable or disable adaptive cache sizing based on memory pressure.
 *
 * When enabled, the cache automatically adjusts its size based on system memory
 * pressure and usage patterns. This provides optimal memory utilization while
 * maintaining performance under varying system conditions.
 *
 * @param dataset Dataset to configure
 * @param enabled Whether to enable adaptive cache sizing
 * @return true on success, false on error
 * @see streaming/streaming-memory-monitor.h for memory monitoring implementation
 */
bool llama_dataset_set_adaptive_cache_sizing(struct llama_dataset * dataset, bool enabled);

/**
 * @brief Get streaming cache statistics and performance metrics.
 *
 * Provides detailed statistics about cache performance, memory usage, and
 * access patterns. These metrics are useful for performance tuning and
 * monitoring dataset access efficiency.
 *
 * @param dataset Dataset to query
 * @param hit_ratio Pointer to store cache hit ratio (0.0-1.0, higher is better)
 * @param memory_usage_bytes Pointer to store current cache memory usage
 * @param entry_count Pointer to store number of cached entries
 * @return true on success, false on error or if streaming is not enabled
 * @see tools/streaming-optimization-analysis.cpp for detailed performance analysis
 */
bool llama_dataset_get_streaming_stats(
    const struct llama_dataset * dataset,
    double * hit_ratio,
    size_t * memory_usage_bytes,
    size_t * entry_count);
#ifdef __cplusplus
}
#endif
