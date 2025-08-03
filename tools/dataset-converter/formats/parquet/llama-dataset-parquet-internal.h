#pragma once

/**
 * @file llama-dataset-parquet-internal.h
 * @brief Internal structures and private interfaces for Parquet dataset handling.
 *
 * This header defines the internal architecture and private implementation details
 * for the Parquet dataset format handler within the llama.cpp dataset converter.
 * It provides the foundational data structures, function declarations, and
 * implementation interfaces that enable efficient Parquet file processing,
 * schema analysis, data conversion, and streaming operations.
 *
 * ## Module Responsibilities
 *
 * The internal Parquet module is responsible for:
 * - **Internal Data Structures**: Defines parquet_format_data and related structures
 *   that maintain state for Parquet processing operations
 * - **Private Function Interfaces**: Declares internal functions for schema analysis,
 *   data conversion, streaming operations, and cache management
 * - **Implementation Coordination**: Provides the interface contracts between
 *   different Parquet implementation files (core, schema, conversion, streaming)
 * - **Memory Management**: Defines interfaces for efficient memory usage,
 *   cache management, and memory pressure handling
 * - **Arrow Integration**: Manages the integration with Apache Arrow libraries
 *   for efficient columnar data processing
 * - **Tokenization Support**: Provides internal interfaces for text tokenization
 *   and mixed content handling (text + pre-tokenized data)
 *
 * ## Architecture Overview
 *
 * The internal structure is organized into several functional areas:
 *
 * ### Core Data Structures
 * - `parquet_format_data`: Main state container for Parquet operations
 * - `parquet_schema_info`: Schema analysis results and metadata
 * - Internal caching structures for tokenization and data access
 *
 * ### Functional Modules
 * - **Schema Analysis** (llama-dataset-parquet-schema.cpp): Column type detection,
 *   mixed content analysis, and schema validation
 * - **Data Conversion** (llama-dataset-parquet-conversion.cpp): Arrow-to-token
 *   conversion, GGUF dataset creation, and data transformation
 * - **Streaming Operations** (llama-dataset-parquet-streaming.cpp): Memory-efficient
 *   data access, streaming optimization, and cache management
 * - **Core Functions** (llama-dataset-parquet-core.cpp): Main loading logic,
 *   metadata extraction, and coordination between modules
 *
 * ## Implementation Details
 *
 * ### Memory Management Strategy
 * The module implements a multi-tier memory management approach:
 * - **Streaming Mode**: For large datasets, data is loaded on-demand
 * - **Tokenization Cache**: LRU cache for frequently accessed text-to-token conversions
 * - **Memory Pressure Handling**: Automatic cache reduction when memory is constrained
 * - **Arrow Integration**: Efficient zero-copy operations where possible
 *
 * ### Mixed Content Support
 * The module supports Parquet files containing both text and pre-tokenized data:
 * - Automatic schema analysis to detect column types
 * - Intelligent column selection based on preferences
 * - Efficient processing of heterogeneous data formats
 * - Seamless integration with the tokenization pipeline
 *
 * ### Performance Optimizations
 * - **Batch Processing**: Efficient batch tokenization for text columns
 * - **Cache Strategies**: Multi-level caching for tokens and metadata
 * - **Streaming Optimization**: Automatic detection of when to enable streaming
 * - **Memory Monitoring**: Proactive memory pressure detection and mitigation
 *
 * ## Usage Guidelines
 *
 * This header should only be included by implementation files (.cpp) within
 * the Parquet format handler. External modules should use the public interface
 * defined in llama-dataset-parquet.h. The internal interfaces are subject to
 * change and should not be relied upon by external code.
 *
 * ## Thread Safety
 *
 * The internal structures and functions are designed to be thread-safe when
 * used correctly. However, concurrent access to the same dataset instance
 * requires external synchronization. The tokenization cache includes internal
 * synchronization for thread-safe operation.
 *
 * ## Error Handling
 *
 * Internal functions follow consistent error handling patterns:
 * - Boolean return values for success/failure indication
 * - Null pointer returns for allocation failures
 * - Detailed error logging through the llama logging system
 * - Graceful degradation when optional features are unavailable
 *
 * @author llama.cpp dataset converter team
 * @version 1.0
 * @since 2024
 *
 * @see llama-dataset-parquet.h for public interface
 * @see core/llama-dataset.h for base dataset functionality
 * @see streaming/streaming-cache.h for streaming infrastructure
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
 * This structure serves as the central state container for all Parquet dataset
 * operations. It maintains the complete context needed for efficient Parquet
 * file processing, including Arrow/Parquet objects, schema analysis results,
 * tokenization support, and caching infrastructure.
 *
 * The structure is designed to support both regular (full-load) and streaming
 * modes of operation, with automatic optimization based on dataset size and
 * available memory. It integrates tightly with the Apache Arrow ecosystem
 * for efficient columnar data processing.
 *
 * ## Memory Layout Considerations
 *
 * The structure is organized to minimize memory fragmentation and optimize
 * cache locality. Frequently accessed fields are grouped together, and
 * large data structures (like caches) are placed at the end.
 *
 * ## Lifecycle Management
 *
 * - **Initialization**: Created during dataset loading with minimal state
 * - **Schema Analysis**: Populated with column type information and preferences
 * - **Data Loading**: Arrow objects and caches are populated as needed
 * - **Streaming Setup**: Additional streaming-specific state is initialized
 * - **Cleanup**: All resources are properly freed through format_data cleanup
 *
 * @note This structure contains both C and C++ members for compatibility.
 *       C++ members are conditionally compiled and have opaque pointer
 *       equivalents for C compatibility.
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
    uint64_t                                    n_sequences;        ///< Total number of sequences in the dataset
    int32_t                                     max_length;         ///< Maximum sequence length across all sequences

    // Schema analysis results
    struct parquet_schema_info                  schema_info;        ///< Detailed schema analysis results
    bool                                        schema_analyzed;    ///< Whether schema analysis has been completed

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
    bool                                        mixed_content;      ///< Whether the dataset contains both text and token columns
};

//
// Forward declarations for internal functions
//

/**
 * @brief Schema analysis functions (implemented in llama-dataset-parquet-schema.cpp)
 *
 * These functions provide comprehensive schema analysis capabilities for Parquet files,
 * enabling automatic detection of column types, mixed content support, and intelligent
 * column selection. The schema analysis is crucial for determining the optimal
 * processing strategy for each dataset.
 *
 * ## Schema Analysis Process
 *
 * 1. **Column Type Detection**: Analyze Arrow data types to classify columns
 * 2. **Content Classification**: Determine if columns contain text or tokens
 * 3. **Mixed Content Analysis**: Detect files with both text and pre-tokenized data
 * 4. **Primary Column Selection**: Choose optimal columns based on preferences
 * 5. **Validation**: Ensure schema compatibility with dataset requirements
 *
 * ## Supported Column Types
 *
 * - **Text Columns**: String, large string, and UTF-8 encoded text data
 * - **Token Columns**: Integer arrays, lists of integers, and numeric sequences
 * - **Mixed Columns**: Columns that may contain both text and numeric data
 *
 * The analysis handles various Arrow data types and provides fallback strategies
 * for ambiguous or complex column structures.
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
 *
 * These functions handle the conversion of data between different formats within
 * the Parquet processing pipeline. They provide efficient transformation of
 * Arrow columnar data to the token-based format required by the dataset API,
 * with support for various data types and optimization strategies.
 *
 * ## Conversion Pipeline
 *
 * 1. **Arrow Array Processing**: Extract data from Arrow arrays with type safety
 * 2. **Data Type Conversion**: Handle various numeric and string types
 * 3. **Token Vector Creation**: Build efficient token sequences
 * 4. **GGUF Integration**: Create GGUF-compatible dataset structures
 * 5. **Memory Optimization**: Minimize copying and memory allocation
 *
 * ## Supported Conversions
 *
 * - **Numeric Arrays**: Direct conversion of integer arrays to token vectors
 * - **String Data**: Text tokenization using llama tokenizer integration
 * - **List Types**: Nested list structures to flattened token sequences
 * - **Mixed Types**: Intelligent handling of heterogeneous data
 *
 * The conversion functions are optimized for both memory efficiency and
 * processing speed, with special handling for large datasets and streaming scenarios.
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
 *
 * These functions implement the streaming infrastructure for memory-efficient
 * processing of large Parquet datasets. The streaming system provides on-demand
 * data loading, intelligent caching, and automatic memory pressure management
 * to handle datasets that exceed available system memory.
 *
 * ## Streaming Architecture
 *
 * 1. **On-Demand Loading**: Data is loaded only when requested
 * 2. **Intelligent Caching**: Frequently accessed data is cached for performance
 * 3. **Memory Monitoring**: Continuous monitoring of memory usage and pressure
 * 4. **Adaptive Optimization**: Dynamic adjustment of cache sizes and strategies
 * 5. **Prefetch Optimization**: Predictive loading based on access patterns
 *
 * ## Memory Management Strategy
 *
 * - **Threshold-Based Activation**: Streaming is enabled based on dataset size
 * - **LRU Cache Management**: Least recently used data is evicted first
 * - **Memory Pressure Response**: Automatic cache reduction under memory pressure
 * - **Batch Processing**: Efficient batch loading to minimize I/O overhead
 *
 * ## Performance Optimizations
 *
 * - **Zero-Copy Operations**: Minimize data copying where possible
 * - **Vectorized Processing**: Leverage SIMD operations for data conversion
 * - **Asynchronous I/O**: Background loading for improved responsiveness
 * - **Cache Locality**: Optimize data layout for CPU cache efficiency
 *
 * The streaming system integrates with the broader dataset converter streaming
 * infrastructure to provide consistent performance characteristics across
 * different data formats.
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
 *
 * These functions provide comprehensive cache management and memory pressure
 * handling capabilities for the Parquet dataset processing pipeline. They
 * implement sophisticated caching strategies, memory monitoring, and adaptive
 * optimization to ensure efficient resource utilization under varying
 * memory conditions.
 *
 * ## Cache Management Strategy
 *
 * 1. **Multi-Level Caching**: Separate caches for tokens, metadata, and raw data
 * 2. **LRU Eviction**: Least recently used items are evicted first
 * 3. **Size-Based Limits**: Configurable cache size limits with automatic enforcement
 * 4. **Hit Ratio Monitoring**: Performance tracking for cache effectiveness
 * 5. **Adaptive Sizing**: Dynamic cache size adjustment based on usage patterns
 *
 * ## Memory Pressure Handling
 *
 * - **Proactive Monitoring**: Continuous monitoring of system memory usage
 * - **Threshold-Based Response**: Automatic cache reduction when thresholds are exceeded
 * - **Graceful Degradation**: Maintain functionality while reducing memory usage
 * - **Recovery Mechanisms**: Automatic cache rebuilding when memory pressure subsides
 *
 * ## Performance Metrics
 *
 * The cache system provides detailed performance metrics including:
 * - Cache hit/miss ratios for performance analysis
 * - Memory usage statistics for capacity planning
 * - Eviction rates for optimization tuning
 * - Access pattern analysis for predictive optimization
 *
 * These functions work in coordination with the broader streaming infrastructure
 * to provide optimal performance across different usage scenarios and system
 * configurations.
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
 *
 * These functions implement the core Parquet dataset loading and management
 * functionality. They provide the main entry points for Parquet file processing,
 * coordinate between different functional modules, and handle the overall
 * dataset lifecycle from loading to cleanup.
 *
 * ## Core Responsibilities
 *
 * 1. **Dataset Loading**: Main entry point for loading Parquet datasets
 * 2. **Module Coordination**: Orchestrate schema analysis, conversion, and streaming
 * 3. **Error Handling**: Comprehensive error detection and recovery
 * 4. **Resource Management**: Proper initialization and cleanup of all resources
 * 5. **Configuration Management**: Handle user preferences and optimization settings
 *
 * ## Loading Process
 *
 * The core loading process follows these steps:
 * 1. **File Validation**: Verify file accessibility and basic format validity
 * 2. **Schema Analysis**: Analyze column types and detect mixed content
 * 3. **Memory Estimation**: Estimate memory requirements for optimization decisions
 * 4. **Mode Selection**: Choose between regular and streaming modes
 * 5. **Data Processing**: Execute the appropriate loading strategy
 * 6. **Optimization Setup**: Configure caching and streaming optimizations
 *
 * ## Integration Points
 *
 * The core functions integrate with:
 * - **Schema Module**: For column analysis and type detection
 * - **Conversion Module**: For data transformation and GGUF creation
 * - **Streaming Module**: For memory-efficient large dataset handling
 * - **Cache System**: For performance optimization and memory management
 * - **Validation System**: For data integrity and format compliance
 *
 * ## Error Recovery
 *
 * The core functions implement robust error recovery mechanisms:
 * - Graceful fallback from streaming to regular mode
 * - Automatic retry with different column selections
 * - Detailed error reporting for debugging and user feedback
 * - Resource cleanup on failure to prevent memory leaks
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
