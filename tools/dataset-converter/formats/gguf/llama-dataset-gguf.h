#pragma once

/**
 * @file llama-dataset-gguf.h
 * @brief GGUF format support module for the llama.cpp dataset converter framework.
 *
 * This module provides comprehensive support for loading, parsing, and streaming GGUF
 * (GPT-Generated Unified Format) datasets. GGUF is the native format for the dataset
 * converter and offers optimal performance, full metadata support, and advanced streaming
 * capabilities with intelligent caching and prefetching.
 *
 * ## GGUF Format Overview
 *
 * GGUF is a binary format designed specifically for machine learning datasets and models.
 * It provides:
 * - **Efficient Storage**: Optimized binary representation with minimal overhead
 * - **Rich Metadata**: Comprehensive metadata support with typed key-value pairs
 * - **Streaming Support**: Native support for on-demand loading and memory-efficient access
 * - **Cross-Platform**: Platform-independent binary format with consistent behavior
 * - **Extensibility**: Forward-compatible design supporting future enhancements
 *
 * ## Key Features
 *
 * ### Native Format Advantages
 * - **Zero-Copy Access**: Direct memory mapping for optimal performance
 * - **Metadata Preservation**: Complete metadata retention during format conversions
 * - **Streaming Optimization**: Purpose-built for streaming access patterns
 * - **Validation Support**: Built-in integrity checking and format validation
 * - **Memory Efficiency**: Minimal memory overhead with intelligent caching
 *
 * ### Streaming Capabilities
 * - **On-Demand Loading**: Tensor data loaded only when accessed
 * - **LRU Caching**: Intelligent caching with configurable memory limits
 * - **Read-Ahead Buffering**: Predictive loading based on access patterns
 * - **Adaptive Sizing**: Dynamic cache adjustment based on memory pressure
 * - **Performance Monitoring**: Detailed statistics and performance metrics
 *
 * ### Metadata Support
 * - **Standard Keys**: Full support for training.* metadata keys
 * - **Custom Metadata**: Extensible metadata with user-defined keys
 * - **Type Safety**: Strongly typed metadata with automatic conversion
 * - **Format Validation**: Automatic validation of metadata consistency
 *
 * ## Usage Patterns
 *
 * ### Basic GGUF Loading
 * ```c
 * // Load GGUF dataset with default settings
 * struct llama_dataset* dataset = llama_dataset_from_gguf(params);
 * 
 * // Access sequences efficiently
 * uint64_t count = llama_dataset_n_sequences(dataset);
 * for (uint64_t i = 0; i < count; i++) {
 *     const int32_t* tokens = llama_dataset_sequence(dataset, i);
 *     // Process tokens...
 * }
 * 
 * llama_dataset_free(dataset);
 * ```
 *
 * ### Advanced Streaming Configuration
 * ```c
 * // Configure streaming for large datasets
 * struct llama_dataset* dataset = llama_dataset_load_gguf(params);
 * 
 * // Set cache size to 100MB for optimal performance
 * llama_dataset_set_streaming_cache_size(dataset, 100 * 1024 * 1024);
 * 
 * // Enable read-ahead with 10-sequence window
 * llama_dataset_set_streaming_read_ahead(dataset, true, 10);
 * 
 * // Enable adaptive cache sizing for memory efficiency
 * llama_dataset_set_adaptive_cache_sizing(dataset, true);
 * ```
 *
 * ### Performance Monitoring
 * ```c
 * // Monitor streaming performance
 * double hit_ratio;
 * size_t memory_usage, entry_count;
 * if (llama_dataset_get_streaming_stats(dataset, &hit_ratio, &memory_usage, &entry_count)) {
 *     printf("Cache hit ratio: %.2f%%, Memory: %zu bytes, Entries: %zu\n",
 *            hit_ratio * 100, memory_usage, entry_count);
 * }
 * ```
 *
 * ## Integration with Core Dataset API
 *
 * This module seamlessly integrates with the core dataset interface defined in
 * llama-dataset.h, providing:
 * - **Transparent Access**: Standard dataset interface with GGUF-specific optimizations
 * - **Metadata Integration**: Full metadata support through standard metadata functions
 * - **Error Handling**: Comprehensive error reporting through standard error interface
 * - **Streaming Integration**: Native streaming support with configurable parameters
 * - **Validation Integration**: Built-in validation through the validation subsystem
 *
 * ## Performance Characteristics
 *
 * ### Memory Usage
 * - **Minimal Overhead**: ~100 bytes per dataset plus cache size
 * - **Configurable Cache**: User-controlled memory usage with LRU eviction
 * - **Adaptive Sizing**: Automatic adjustment based on system memory pressure
 * - **Zero-Copy Access**: Direct memory mapping when possible
 *
 * ### Access Patterns
 * - **Sequential Access**: Optimal performance with read-ahead buffering
 * - **Random Access**: Efficient with appropriate cache sizing
 * - **Mixed Patterns**: Adaptive optimization based on detected patterns
 * - **Large Datasets**: Streaming support for datasets larger than available memory
 *
 * ## Implementation Details
 *
 * ### Internal Architecture
 * - **GGUF Context**: Native GGUF context management with automatic cleanup
 * - **GGML Integration**: Seamless integration with GGML tensor operations
 * - **Streaming Cache**: LRU cache with configurable size and adaptive behavior
 * - **Memory Mapping**: Platform-specific memory mapping for optimal performance
 * - **Error Recovery**: Robust error handling with detailed diagnostic information
 *
 * ### Thread Safety
 * - **Read Operations**: Thread-safe for concurrent read access
 * - **Cache Management**: Thread-safe cache operations with minimal locking
 * - **Configuration**: Configuration changes require external synchronization
 * - **Error State**: Thread-local error state for multi-threaded environments
 *
 * ## Format Validation
 *
 * The module provides comprehensive validation including:
 * - **Header Validation**: GGUF header structure and version checking
 * - **Metadata Validation**: Metadata consistency and type validation
 * - **Tensor Validation**: Tensor structure and data integrity checking
 * - **Cross-Reference Validation**: Consistency between metadata and tensor data
 *
 * @see core/llama-dataset.h for the core dataset interface
 * @see streaming/streaming-cache.h for streaming implementation details
 * @see validation/llama-dataset-validation.h for validation capabilities
 * @see tools/convert-to-gguf.cpp for GGUF conversion utilities
 * @see formats/gguf/llama-dataset-gguf-utils.h for GGUF utility functions
 *
 * @version 1.0
 * @since 2024
 */

#include "../../core/llama-dataset.h"

// Forward declarations for GGUF and GGML integration
struct gguf_context;
struct ggml_context;
struct IFormatLoader;

#ifdef __cplusplus
extern "C" {
#endif

//
// GGUF-Specific Streaming Interface
//
// These functions provide GGUF-specific streaming capabilities that leverage
// the native format advantages for optimal performance and memory efficiency.
//

/**
 * @brief Get tensor data from a GGUF file in streaming mode.
 *
 * This function provides on-demand loading of tensor data from GGUF files with
 * intelligent caching and memory management. It is the core function used internally
 * by the sequence access functions to implement streaming behavior.
 *
 * ## Streaming Behavior
 *
 * The function implements sophisticated streaming logic:
 * - **Cache Lookup**: First checks the LRU cache for previously loaded data
 * - **On-Demand Loading**: Loads tensor data from file only when not cached
 * - **Memory Management**: Automatically manages memory allocation and cleanup
 * - **Error Recovery**: Provides robust error handling with detailed diagnostics
 * - **Performance Optimization**: Optimizes access patterns for sequential and random access
 *
 * ## Caching Strategy
 *
 * The caching implementation uses:
 * - **LRU Eviction**: Least Recently Used eviction policy for optimal cache utilization
 * - **Adaptive Sizing**: Dynamic cache size adjustment based on memory pressure
 * - **Prefetching**: Optional read-ahead buffering for sequential access patterns
 * - **Memory Monitoring**: Continuous monitoring of system memory availability
 * - **Cache Statistics**: Detailed performance metrics for optimization
 *
 * ## Performance Characteristics
 *
 * - **Cache Hit**: ~10-50 nanoseconds (direct memory access)
 * - **Cache Miss**: ~1-10 milliseconds (file I/O + decompression if applicable)
 * - **Memory Overhead**: ~64 bytes per cached tensor plus tensor data size
 * - **Scalability**: Supports datasets from MB to TB scale with constant memory usage
 *
 * ## Error Handling
 *
 * The function provides comprehensive error handling:
 * - **Invalid Parameters**: Validates dataset pointer and index bounds
 * - **File I/O Errors**: Handles file access errors with detailed error messages
 * - **Memory Allocation**: Graceful handling of memory allocation failures
 * - **Format Errors**: Detects and reports GGUF format inconsistencies
 * - **Cache Errors**: Handles cache-related errors with automatic recovery
 *
 * ## Thread Safety
 *
 * This function is thread-safe for concurrent read access:
 * - **Cache Access**: Thread-safe cache operations with minimal locking
 * - **File I/O**: Concurrent file access with proper synchronization
 * - **Memory Management**: Thread-safe memory allocation and deallocation
 * - **Error State**: Thread-local error reporting for multi-threaded environments
 *
 * @param dataset Dataset to query (must be a valid GGUF dataset)
 * @param index Index of the tensor to load (must be < llama_dataset_n_sequences())
 * @return Pointer to the tensor data on success, NULL on error
 *
 * @note The returned pointer is valid until the dataset is freed or the cache
 *       evicts the entry. For long-term storage, copy the data to user-managed memory.
 * @note This function is primarily for internal use. User code should prefer
 *       llama_dataset_sequence() for standard sequence access.
 * @note Error details can be retrieved using llama_dataset_get_error_message()
 *       and llama_dataset_get_error_code() after a NULL return.
 *
 * @see llama_dataset_sequence() for the standard sequence access interface
 * @see llama_dataset_set_streaming_cache_size() for cache configuration
 * @see llama_dataset_get_streaming_stats() for performance monitoring
 * @see streaming/streaming-cache.h for detailed cache implementation
 *
 * @warning The returned pointer may become invalid if the cache evicts the entry.
 *          Do not store the pointer for extended periods without copying the data.
 */
void * llama_dataset_gguf_get_tensor_data_streaming(const struct llama_dataset * dataset, uint64_t index);

/**
 * @brief Load a dataset from a GGUF file with advanced streaming and configuration options.
 *
 * This function provides the primary implementation for loading GGUF datasets with
 * comprehensive support for both standard and streaming modes. It handles the complete
 * lifecycle of GGUF dataset initialization including format validation, context creation,
 * memory management, and streaming configuration.
 *
 * ## Loading Process Overview
 *
 * The loading process follows a carefully orchestrated sequence:
 * 1. **Parameter Validation**: Validates input parameters and file accessibility
 * 2. **Format Validation**: Comprehensive GGUF format validation and integrity checking
 * 3. **Context Creation**: Creates GGUF and GGML contexts based on loading mode
 * 4. **Tensor Management**: Configures tensor loading strategy (immediate vs. on-demand)
 * 5. **Cache Initialization**: Sets up streaming cache and optimization components
 * 6. **Metadata Extraction**: Extracts and caches GGUF metadata for efficient access
 *
 * ## Loading Modes
 *
 * ### Standard Mode (dataset_streaming = false)
 * - **Memory Strategy**: All tensor data loaded into memory during initialization
 * - **Performance**: Optimal access speed (~10-50 nanoseconds per sequence)
 * - **Memory Usage**: Full dataset size in memory
 * - **Use Case**: Small to medium datasets where memory usage is not a concern
 * - **GGML Integration**: Full GGML context with all tensors allocated
 *
 * ### Streaming Mode (dataset_streaming = true)
 * - **Memory Strategy**: Tensor data loaded on-demand with intelligent caching
 * - **Performance**: Cache hits ~10-50 ns, cache misses ~1-10 ms
 * - **Memory Usage**: Configurable cache size (default: adaptive)
 * - **Use Case**: Large datasets or memory-constrained environments
 * - **GGML Integration**: Lightweight context with metadata-only loading
 *
 * @param common_params Common parameters including file path and streaming configuration
 * @return Pointer to the dataset, or NULL on error
 *
 * @note This function integrates with the streaming subsystem for optimal performance
 * @note Error details can be retrieved using llama_dataset_get_error_message()
 * @note The returned dataset must be freed using llama_dataset_free()
 *
 * @see llama_dataset_from_gguf() for the public factory function
 * @see streaming/streaming-cache.h for streaming implementation details
 * @see validation/llama-dataset-validation.h for validation capabilities
 */
struct llama_dataset * llama_dataset_load_gguf(const common_params * common_params);

/**
 * @brief Get the GGUF format loader interface.
 *
 * Returns the IFormatLoader interface implementation for GGUF format.
 * Used by the registry system for format detection and loading.
 *
 * @return GGUF format loader interface
 */
const struct IFormatLoader* gguf_get_loader_interface(void);

#ifdef __cplusplus
}
#endif
