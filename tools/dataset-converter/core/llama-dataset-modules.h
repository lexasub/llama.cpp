#pragma once

/**
 * @file llama-dataset-modules.h
 * @brief Internal module interfaces for the dataset converter framework.
 *
 * This header defines the internal interfaces between all modules in the dataset converter
 * framework. It serves as the central contract definition for module communication,
 * providing function signatures for error handling, metadata access, streaming operations,
 * format loading, and conversion operations.
 *
 * ## Architecture Overview
 *
 * The module interface architecture follows a layered design with clear dependencies:
 *
 * ```
 * ┌─────────────────────────────────────────────────────────────┐
 * │                 Core Coordination Layer                     │
 * │              (llama-dataset-core.cpp)                       │
 * └─────────────────────────────────────────────────────────────┘
 *                              │
 * ┌─────────────────────────────────────────────────────────────┐
 * │                  Functional Modules                         │
 * │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
 * │  │   Error     │ │  Metadata   │ │  Streaming  │           │
 * │  │  Handling   │ │   Access    │ │ Operations  │           │
 * │  └─────────────┘ └─────────────┘ └─────────────┘           │
 * └─────────────────────────────────────────────────────────────┘
 *                              │
 * ┌─────────────────────────────────────────────────────────────┐
 * │                Format-Specific Loaders                      │
 * │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
 * │  │    GGUF     │ │    Text     │ │   Parquet   │           │
 * │  │   Loader    │ │   Loader    │ │   Loader    │           │
 * │  └─────────────┘ └─────────────┘ └─────────────┘           │
 * └─────────────────────────────────────────────────────────────┘
 * ```
 *
 * ## Module Interface Design Principles
 *
 * ### 1. Clear Separation of Concerns
 * Each module has a single, well-defined responsibility:
 * - **Error Module**: Centralized error state management and reporting
 * - **Metadata Module**: Dataset metadata extraction and type conversion
 * - **Streaming Module**: Cache management and streaming optimization
 * - **Format Loaders**: Format-specific loading and initialization
 * - **Conversion Module**: Dataset format transformation operations
 *
 * ### 2. Dependency Minimization
 * Module dependencies are carefully managed:
 * - **Base Module**: Error handling has no dependencies
 * - **Functional Modules**: Depend only on error handling
 * - **Format Loaders**: Depend on error handling and metadata
 * - **Core Coordination**: Orchestrates all modules
 *
 * ### 3. Error Propagation Strategy
 * All modules use consistent error handling:
 * - **Thread-Local Storage**: Error state is thread-local for safety
 * - **Consistent Reporting**: All modules use the same error interface
 * - **Rich Context**: Error messages include module and operation context
 * - **Graceful Cleanup**: Automatic resource cleanup on error conditions
 *
 * @warning This header contains internal interfaces that may change between versions.
 *          Only core dataset converter modules should include this header.
 *
 * @see llama-dataset.h for the public API interface
 * @see llama-dataset-internal.h for internal data structures
 * @see Requirements 1.2, 7.3 for interface specifications
 *
 * @version 1.0
 * @since 2024
 */

#include "llama-dataset.h"
#include "llama-dataset-internal.h"
#include "common.h"
#include "gguf.h"
#include "ggml.h"
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// ERROR HANDLING MODULE INTERFACE
// =============================================================================

/**
 * @brief Internal error management interface for centralized error handling.
 *
 * This interface provides thread-safe error state management across all modules
 * in the dataset converter framework. It uses thread-local storage to maintain
 * independent error state for each thread, enabling safe concurrent operations.
 *
 * @note All error handling functions are thread-safe and use thread-local storage.
 */

/**
 * @brief Set error message with automatic context formatting.
 *
 * Sets the thread-local error state with a formatted error message. The message
 * is automatically truncated if it exceeds the internal buffer size (1024 bytes).
 * This function is thread-safe and can be called from any module.
 *
 * @param message Error message string (will be copied to thread-local storage)
 *
 * @note The message is copied to thread-local storage, so the original string
 *       can be safely freed after this call.
 * @note If message is NULL, sets a generic "Unknown error" message.
 * @note Messages longer than 1023 characters are truncated with "..." suffix.
 */
void llama_dataset_error_set(const char* message);

/**
 * @brief Set error with specific error code and message.
 *
 * Sets the thread-local error state with both an error code and message.
 * This provides both programmatic error handling (through the code) and
 * human-readable error information (through the message).
 *
 * @param code Standardized error code from dataset_error enum
 * @param message Human-readable error description
 *
 * @note The error code can be retrieved with llama_dataset_get_error_code()
 * @note The message follows the same truncation rules as llama_dataset_error_set()
 * @note Setting DATASET_SUCCESS as the code is equivalent to clearing the error
 */
void llama_dataset_error_set_with_code(enum dataset_error code, const char* message);

/**
 * @brief Set error with full context information.
 *
 * Sets the thread-local error state with complete context information including
 * the module name, operation name, error code, and detailed message. This is
 * the most comprehensive error reporting function and should be used by all
 * internal modules for consistent error reporting.
 *
 * @param module Name of the module where the error occurred (e.g., "metadata", "streaming")
 * @param operation Name of the operation that failed (e.g., "get_string", "load_cache")
 * @param code Standardized error code from dataset_error enum
 * @param message Detailed error description with specific context
 *
 * @note The final error message format is: "[module:operation] message"
 * @note All parameters are copied to thread-local storage
 * @note NULL parameters are handled gracefully with default values
 * @note This is the preferred error reporting function for internal modules
 */
void llama_dataset_error_set_with_context(const char* module, const char* operation, 
                                          enum dataset_error code, const char* message);

/**
 * @brief Check if an error has occurred in the current thread.
 *
 * Checks the thread-local error state to determine if any error has been set.
 * This function is used internally by modules to check for error conditions
 * before proceeding with operations.
 *
 * @return true if an error has been set, false otherwise
 *
 * @note This function is thread-safe and only checks the current thread's error state
 * @note Returns false if the error state has been cleared
 * @note This is an internal function; external code should use llama_dataset_has_error()
 */
bool llama_dataset_error_has_error_internal(void);

/**
 * @brief Clear the error state for the current thread.
 *
 * Clears the thread-local error state, resetting both the error code to
 * DATASET_SUCCESS and clearing the error message. This function should be
 * called after handling an error condition to reset the error state.
 *
 * @note This function only affects the current thread's error state
 * @note After calling this function, llama_dataset_error_has_error_internal() will return false
 * @note This is safe to call even if no error is currently set
 * @note Internal modules should call this after handling error conditions
 */
void llama_dataset_error_clear_internal(void);

/**
 * @brief Get the current error message for the current thread.
 *
 * Retrieves the thread-local error message string. The returned string is
 * valid until the next error is set or the error state is cleared.
 *
 * @return Pointer to the error message string, or NULL if no error is set
 *
 * @note The returned pointer is valid until the next error operation
 * @note Returns NULL if no error has been set or if the error has been cleared
 * @note The returned string should not be modified or freed
 * @note This is an internal function; external code should use llama_dataset_get_error()
 */
const char* llama_dataset_error_get_message_internal(void);

/**
 * @brief Get the current error code for the current thread.
 *
 * Retrieves the thread-local error code. This provides programmatic access
 * to the specific type of error that occurred.
 *
 * @return Error code from dataset_error enum, or DATASET_SUCCESS if no error
 *
 * @note Returns DATASET_SUCCESS if no error has been set
 * @note The error code is set by llama_dataset_error_set_with_code() or
 *       llama_dataset_error_set_with_context()
 * @note This is an internal function; external code should use llama_dataset_get_error_code()
 */
enum dataset_error llama_dataset_error_get_code_internal(void);

// =============================================================================
// METADATA ACCESS MODULE INTERFACE
// =============================================================================

/**
 * @brief Internal metadata operations interface for dataset metadata access.
 *
 * This interface provides format-agnostic metadata access across all dataset
 * formats. It handles type conversion, key normalization, and provides
 * consistent metadata access patterns regardless of the underlying format.
 *
 * ## Metadata Architecture
 *
 * The metadata system provides:
 * - **Format Abstraction**: Unified access across GGUF, text, and Parquet formats
 * - **Type Safety**: Robust type checking and conversion for metadata values
 * - **Key Normalization**: Consistent key naming across different source formats
 * - **Default Handling**: Graceful fallback for missing or invalid metadata
 *
 * @note All metadata functions are thread-safe for read operations
 * @note Metadata is cached for performance after first access
 */

/**
 * @brief Get string metadata value with format-specific key mapping.
 *
 * Retrieves a string metadata value from the dataset, handling format-specific
 * key mapping and normalization. The function searches for the key in the
 * format-specific metadata store and returns the value as a string.
 *
 * @param dataset Dataset instance (must not be NULL)
 * @param key Metadata key name (normalized or format-specific)
 * @return String value, or NULL if key not found or dataset is NULL
 *
 * @note The returned string is owned by the dataset and should not be freed
 * @note Returns NULL if the key is not found or the value cannot be converted to string
 * @note Key lookup is case-insensitive and handles format-specific variations
 * @note For GGUF datasets, searches both standard and custom metadata keys
 */
const char* llama_dataset_metadata_get_str_internal(const struct llama_dataset* dataset, const char* key);

/**
 * @brief Get integer metadata value with type conversion and default handling.
 *
 * Retrieves an integer metadata value from the dataset, performing automatic
 * type conversion if necessary. If the key is not found or conversion fails,
 * returns the specified default value.
 *
 * @param dataset Dataset instance (must not be NULL)
 * @param key Metadata key name (normalized or format-specific)
 * @param default_value Value to return if key not found or conversion fails
 * @return Integer value, or default_value if not found/convertible
 *
 * @note Performs automatic type conversion from string and floating-point values
 * @note Returns default_value if dataset is NULL, key is NULL, or key not found
 * @note For string values, attempts to parse as integer (supports decimal, hex, octal)
 * @note For floating-point values, truncates to integer (with range checking)
 * @note Sets error state if conversion fails due to invalid format
 */
int64_t llama_dataset_metadata_get_int_internal(const struct llama_dataset* dataset, const char* key, int64_t default_value);

/**
 * @brief Get floating-point metadata value with precision handling.
 *
 * Retrieves a floating-point metadata value from the dataset, handling
 * different precision formats and performing automatic type conversion.
 * Supports both single and double precision values.
 *
 * @param dataset Dataset instance (must not be NULL)
 * @param key Metadata key name (normalized or format-specific)
 * @param default_value Value to return if key not found or conversion fails
 * @return Floating-point value, or default_value if not found/convertible
 *
 * @note Handles both single-precision (float) and double-precision (double) values
 * @note Performs automatic conversion from string and integer values
 * @note Returns default_value if dataset is NULL, key is NULL, or key not found
 * @note For string values, supports scientific notation and various formats
 * @note For integer values, converts to floating-point with full precision
 * @note Sets error state if conversion fails due to invalid format
 */
float llama_dataset_metadata_get_float_internal(const struct llama_dataset* dataset, const char* key, float default_value);

/**
 * @brief Check if a metadata key exists in the dataset.
 *
 * Checks whether a specific metadata key exists in the dataset without
 * retrieving its value. This is useful for conditional metadata access
 * and validation of dataset completeness.
 *
 * @param dataset Dataset instance (must not be NULL)
 * @param key Metadata key name to check
 * @return true if key exists, false otherwise
 *
 * @note Returns false if dataset is NULL or key is NULL
 * @note Key lookup is case-insensitive and handles format-specific variations
 * @note Does not set error state for missing keys (this is not an error condition)
 * @note More efficient than retrieving the value when only existence matters
 */
bool llama_dataset_metadata_has_key_internal(const struct llama_dataset* dataset, const char* key);

/**
 * @brief Get the number of metadata entries in the dataset.
 *
 * Returns the total number of metadata key-value pairs available in the
 * dataset. This includes both standard metadata and format-specific
 * custom metadata.
 *
 * @param dataset Dataset instance (must not be NULL)
 * @return Number of metadata entries, or 0 if dataset is NULL
 *
 * @note Count includes both standard and custom metadata entries
 * @note For GGUF datasets, includes both GGUF metadata and custom fields
 * @note For text datasets, includes file metadata and processing parameters
 * @note For Parquet datasets, includes schema metadata and column information
 * @note Does not set error state (0 is a valid count for datasets without metadata)
 */
size_t llama_dataset_metadata_get_count_internal(const struct llama_dataset* dataset);

// =============================================================================
// STREAMING OPERATIONS MODULE INTERFACE
// =============================================================================

/**
 * @brief Internal streaming management interface for cache and optimization.
 *
 * This interface provides streaming functionality and cache management for
 * efficient access to large datasets. It includes LRU caching, read-ahead
 * buffering, adaptive optimization, and performance monitoring.
 *
 * ## Streaming Architecture
 *
 * The streaming system consists of:
 * - **LRU Cache**: Intelligent caching with configurable size limits
 * - **Read-Ahead Buffer**: Predictive loading based on access patterns
 * - **Optimization Manager**: Adaptive performance tuning
 * - **Memory Monitor**: System memory pressure detection
 *
 * @note All streaming functions are thread-safe for concurrent access
 * @note Streaming is only available for datasets loaded with streaming=true
 */

/**
 * @brief Set cache size for streaming datasets with memory validation.
 *
 * Configures the cache size for streaming datasets, with automatic validation
 * of memory availability and system constraints. The cache size affects both
 * memory usage and access performance.
 *
 * @param dataset Dataset instance (must be streaming-enabled)
 * @param cache_size_bytes Cache size in bytes (must be > 0)
 * @return true if cache size was set successfully, false on error
 *
 * @note Only works with datasets loaded in streaming mode
 * @note Cache size is validated against available system memory
 * @note Minimum cache size is enforced (typically 1MB)
 * @note Maximum cache size is limited by system memory and configuration
 * @note Setting a new cache size may trigger cache reorganization
 * @note Sets error state if validation fails or dataset is not streaming
 */
bool llama_dataset_streaming_set_cache_size_internal(struct llama_dataset* dataset, size_t cache_size_bytes);

/**
 * @brief Configure read-ahead buffering for streaming optimization.
 *
 * Enables or disables read-ahead buffering and configures the read-ahead
 * window size. Read-ahead buffering can significantly improve performance
 * for sequential access patterns.
 *
 * @param dataset Dataset instance (must be streaming-enabled)
 * @param enabled Whether to enable read-ahead buffering
 * @param window_size Number of sequences to prefetch (ignored if enabled=false)
 * @return true if read-ahead was configured successfully, false on error
 *
 * @note Only works with datasets loaded in streaming mode
 * @note Window size is validated against cache size and memory constraints
 * @note Read-ahead is most effective for sequential access patterns
 * @note Large window sizes may increase memory usage significantly
 * @note Window size of 0 disables read-ahead even if enabled=true
 * @note Sets error state if configuration fails or dataset is not streaming
 */
bool llama_dataset_streaming_set_read_ahead_internal(struct llama_dataset* dataset, bool enabled, size_t window_size);

/**
 * @brief Get comprehensive streaming statistics and performance metrics.
 *
 * Retrieves detailed statistics about streaming performance, including cache
 * hit ratios, memory usage, and optimization effectiveness. This information
 * is useful for performance tuning and monitoring.
 *
 * @param dataset Dataset instance (must be streaming-enabled)
 * @param hit_ratio Output parameter for cache hit ratio (0.0 to 1.0)
 * @param memory_usage_bytes Output parameter for current memory usage
 * @param entry_count Output parameter for number of cached entries
 * @return true if statistics were retrieved successfully, false on error
 *
 * @note All output parameters can be NULL if the corresponding statistic is not needed
 * @note Hit ratio is calculated as (cache_hits / total_accesses)
 * @note Memory usage includes both cached data and cache overhead
 * @note Entry count reflects the current number of cached sequences
 * @note Statistics are reset when cache is cleared or resized
 * @note Sets error state if dataset is not streaming or statistics unavailable
 */
bool llama_dataset_streaming_get_stats_internal(const struct llama_dataset* dataset, 
                                               double* hit_ratio, 
                                               size_t* memory_usage_bytes, 
                                               size_t* entry_count);

/**
 * @brief Reset streaming statistics and performance counters.
 *
 * Resets all streaming statistics to their initial values. This is useful
 * for measuring performance over specific time periods or after configuration
 * changes.
 *
 * @param dataset Dataset instance (must be streaming-enabled)
 * @return true if statistics were reset successfully, false on error
 *
 * @note Resets cache hit counters, access counters, and timing statistics
 * @note Does not affect cached data or cache configuration
 * @note Statistics begin accumulating immediately after reset
 * @note Sets error state if dataset is not streaming
 */
bool llama_dataset_streaming_reset_stats_internal(struct llama_dataset* dataset);

/**
 * @brief Clear streaming cache and free cached memory.
 *
 * Clears all cached data from the streaming cache, freeing the associated
 * memory. This can be useful for memory management or when switching to
 * different access patterns.
 *
 * @param dataset Dataset instance (must be streaming-enabled)
 * @return true if cache was cleared successfully, false on error
 *
 * @note Frees all cached sequence data and resets cache statistics
 * @note Does not change cache configuration (size, read-ahead settings)
 * @note Subsequent accesses will reload data from the underlying storage
 * @note Thread-safe operation that coordinates with concurrent access
 * @note Sets error state if dataset is not streaming
 */
bool llama_dataset_streaming_clear_cache_internal(struct llama_dataset* dataset);

// =============================================================================
// FORMAT LOADER MODULE INTERFACES
// =============================================================================

/**
 * @brief Format-specific loader interfaces for dataset initialization.
 *
 * These interfaces provide format-specific loading functionality for different
 * dataset formats. Each loader is responsible for format-specific parsing,
 * validation, and initialization while providing a consistent interface to
 * the core coordination layer.
 *
 * ## Loader Architecture
 *
 * Each format loader provides:
 * - **Format Detection**: Automatic format validation and detection
 * - **Resource Initialization**: Setup of format-specific contexts and resources
 * - **Metadata Extraction**: Format-specific metadata parsing and normalization
 * - **Error Handling**: Comprehensive error reporting with format-specific context
 *
 * @note All loader functions return NULL on error and set appropriate error state
 * @note Loaders are responsible for complete resource cleanup on failure
 * @note All loaders support both streaming and non-streaming modes
 */

/**
 * @brief Load GGUF format dataset with native optimization.
 *
 * Loads a dataset from GGUF format file, providing native performance and
 * full feature support. GGUF is the native format for the dataset converter
 * framework and offers optimal performance characteristics.
 *
 * @param params Common parameters including file path and streaming configuration
 * @return Pointer to loaded dataset, or NULL on error
 *
 * @note GGUF format provides the best performance and lowest memory overhead
 * @note Supports all metadata types and custom metadata fields
 * @note Native streaming support with optimal cache efficiency
 * @note Automatic validation of GGUF format version and compatibility
 * @note Sets detailed error state on failure with GGUF-specific context
 */
struct llama_dataset* llama_dataset_gguf_loader_load(const common_params* params);

/**
 * @brief Load text format dataset with tokenization support.
 *
 * Loads a dataset from text format file, performing tokenization using the
 * provided llama model. Supports various text formats and encoding schemes
 * with intelligent tokenization caching for performance.
 *
 * @param params Common parameters including file path and streaming configuration
 * @param model Llama model for tokenization (must not be NULL)
 * @return Pointer to loaded dataset, or NULL on error
 *
 * @note Requires a valid llama model for tokenization operations
 * @note Supports streaming mode with on-demand tokenization
 * @note Automatic encoding detection (UTF-8, UTF-16, etc.)
 * @note Intelligent tokenization caching for repeated access
 * @note Sets detailed error state on failure with text-specific context
 */
struct llama_dataset* llama_dataset_text_loader_load(const common_params* params, struct llama_model* model);

/**
 * @brief Load Parquet format dataset with Apache Arrow integration.
 *
 * Loads a dataset from Parquet format file using Apache Arrow for efficient
 * columnar data access. Supports schema analysis, column selection, and
 * batch processing for optimal performance with large datasets.
 *
 * @param params Common parameters including file path and streaming configuration
 * @return Pointer to loaded dataset, or NULL on error
 *
 * @note Requires Apache Arrow/Parquet support to be compiled in
 * @note Supports streaming mode with efficient columnar access
 * @note Automatic schema analysis and column type detection
 * @note Configurable batch processing for memory efficiency
 * @note Sets detailed error state on failure with Parquet-specific context
 */
struct llama_dataset* llama_dataset_parquet_loader_load(const common_params* params);

/**
 * @brief Load Parquet format dataset with integrated tokenization.
 *
 * Loads a dataset from Parquet format file and performs tokenization using
 * the provided llama model. This combines Parquet loading with text processing
 * for datasets that contain text data in columnar format.
 *
 * @param params Common parameters including file path and streaming configuration
 * @param model Llama model for tokenization (must not be NULL)
 * @return Pointer to loaded dataset, or NULL on error
 *
 * @note Combines Parquet loading with tokenization in a single operation
 * @note Requires both Apache Arrow/Parquet support and a valid llama model
 * @note Supports streaming mode with on-demand tokenization
 * @note Automatic text column detection and processing
 * @note Intelligent tokenization caching for columnar data
 * @note Sets detailed error state on failure with combined context information
 */
struct llama_dataset* llama_dataset_parquet_loader_load_with_tokenization(const common_params* params, struct llama_model* model);

// =============================================================================
// CONVERSION OPERATIONS MODULE INTERFACE
// =============================================================================

/**
 * @brief Dataset format conversion interface for format transformation.
 *
 * This interface provides comprehensive dataset format conversion capabilities,
 * allowing transformation between different dataset formats while preserving
 * metadata, sequence structure, and optimization characteristics.
 *
 * ## Conversion Architecture
 *
 * The conversion system provides:
 * - **Format Transformation**: Convert between GGUF, text, and Parquet formats
 * - **Metadata Preservation**: Maintain metadata across format conversions
 * - **Optimization Transfer**: Preserve streaming and cache configurations
 * - **Validation Integration**: Comprehensive validation of conversion results
 *
 * @note All conversion functions validate input parameters and set error state on failure
 * @note Conversions preserve streaming configuration when supported by target format
 * @note Source dataset remains unchanged during conversion operations
 */

/**
 * @brief Convert dataset to GGUF format with optimization preservation.
 *
 * Converts a dataset from any supported format to GGUF format, preserving
 * all metadata, sequence structure, and optimization configurations. GGUF
 * is the native format and supports all features of the dataset framework.
 *
 * @param source_dataset Source dataset to convert (must not be NULL)
 * @param output_path Path for the output GGUF file
 * @return Pointer to new GGUF dataset, or NULL on error
 *
 * @note Source dataset remains unchanged and valid after conversion
 * @note All metadata is preserved with format-appropriate key mapping
 * @note Streaming configuration is preserved in the new dataset
 * @note Output file is created with appropriate permissions and validation
 * @note Sets detailed error state on failure with conversion-specific context
 */
struct llama_dataset* llama_dataset_conversion_to_gguf(const struct llama_dataset* source_dataset, const char* output_path);

/**
 * @brief Convert GGUF dataset to specified target format.
 *
 * Converts a GGUF dataset to the specified target format, handling format-specific
 * requirements and limitations. This is useful for exporting datasets to formats
 * compatible with other tools and frameworks.
 *
 * @param gguf_dataset Source GGUF dataset (must not be NULL)
 * @param target_format Target format for conversion
 * @param output_path Path for the output file
 * @param model Tokenization model (required for text format, can be NULL for others)
 * @return Pointer to new dataset in target format, or NULL on error
 *
 * @note Source GGUF dataset remains unchanged after conversion
 * @note Some metadata may be lost if not supported by target format
 * @note Streaming configuration is preserved when supported by target format
 * @note Model parameter is required for text format conversion, ignored for others
 * @note Sets detailed error state on failure with format-specific context
 */
struct llama_dataset* llama_dataset_conversion_from_gguf(const struct llama_dataset* gguf_dataset, 
                                                        enum dataset_type target_format, 
                                                        const char* output_path, 
                                                        struct llama_model* model);

/**
 * @brief Validate conversion result against source dataset.
 *
 * Performs comprehensive validation of a conversion result by comparing
 * the converted dataset against the original source dataset. This ensures
 * data integrity and conversion accuracy.
 *
 * @param source_dataset Original source dataset
 * @param converted_dataset Result of conversion operation
 * @param validation_level Level of validation to perform (basic, comprehensive, etc.)
 * @return true if validation passes, false if discrepancies found
 *
 * @note Validation includes sequence count, metadata preservation, and data integrity
 * @note Higher validation levels perform more thorough but slower checks
 * @note Sets detailed error state if validation fails with specific discrepancy information
 * @note Both datasets must be valid and accessible during validation
 */
bool llama_dataset_conversion_validate(const struct llama_dataset* source_dataset, 
                                      const struct llama_dataset* converted_dataset, 
                                      int validation_level);

// =============================================================================
// RESOURCE MANAGEMENT MODULE INTERFACE
// =============================================================================

/**
 * @brief Internal resource management interface for dataset lifecycle.
 *
 * This interface provides internal resource management functions for dataset
 * allocation, initialization, and cleanup. These functions are used by the
 * core coordination layer and format loaders for consistent resource handling.
 *
 * @note These functions are for internal use by dataset converter modules only
 * @note All functions handle NULL parameters gracefully
 * @note Resource cleanup is automatic and comprehensive
 */

/**
 * @brief Allocate and initialize dataset structure for specific format.
 *
 * Allocates a new dataset structure and performs basic initialization for
 * the specified format and streaming configuration. This is the standard
 * allocation function used by all format loaders.
 *
 * @param type Dataset format type
 * @param streaming Whether to enable streaming mode
 * @return Pointer to allocated dataset, or NULL on error
 *
 * @note Initializes all structure members to safe default values
 * @note Sets up streaming infrastructure if streaming=true
 * @note Does not perform format-specific initialization (done by loaders)
 * @note Sets error state on allocation failure
 * @note Returned dataset must be freed with llama_dataset_free_internal()
 */
struct llama_dataset* llama_dataset_alloc_internal(enum dataset_type type, bool streaming);

/**
 * @brief Free dataset and all associated resources.
 *
 * Performs comprehensive cleanup of a dataset structure and all associated
 * resources, including format-specific data, streaming cache, and contexts.
 * This function coordinates cleanup across all modules.
 *
 * @param dataset Dataset to free (can be NULL)
 *
 * @note Safe to call with NULL dataset pointer
 * @note Coordinates cleanup across all modules (streaming, format-specific, etc.)
 * @note Handles partial initialization (safe to call on failed allocations)
 * @note Does not set error state (cleanup should always succeed)
 * @note After calling this function, the dataset pointer is invalid
 */
void llama_dataset_free_internal(struct llama_dataset* dataset);

/**
 * @brief Initialize streaming infrastructure for dataset.
 *
 * Sets up streaming cache, optimization manager, and related infrastructure
 * for a dataset that was allocated with streaming=true. This function is
 * called by format loaders after basic dataset setup.
 *
 * @param dataset Dataset to initialize streaming for (must be streaming-enabled)
 * @param initial_cache_size Initial cache size in bytes
 * @return true if streaming was initialized successfully, false on error
 *
 * @note Only works with datasets allocated with streaming=true
 * @note Sets up LRU cache, optimization manager, and memory monitoring
 * @note Initial cache size is validated against system memory constraints
 * @note Sets error state if initialization fails
 * @note Streaming infrastructure is automatically cleaned up by llama_dataset_free_internal()
 */
bool llama_dataset_init_streaming_internal(struct llama_dataset* dataset, size_t initial_cache_size);

#ifdef __cplusplus
}
#endif