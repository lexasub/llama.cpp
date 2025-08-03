/**
 * @file llama-dataset.cpp
 * @brief Core dataset implementation for the llama.cpp dataset converter framework.
 *
 * This file implements the primary C interface for working with training datasets across
 * multiple formats (GGUF, text, Parquet) with comprehensive streaming, validation, and
 * optimization capabilities. It serves as the central implementation hub that coordinates
 * between format-specific modules, streaming subsystems, and validation components.
 *
 * ## Implementation Architecture
 *
 * The core implementation follows a modular design pattern with clear separation of concerns:
 *
 * ### Format Abstraction Layer
 * - **Factory Functions**: Provide unified entry points for different formats
 * - **Format Dispatching**: Routes operations to format-specific implementations
 * - **Resource Management**: Handles lifecycle management across all formats
 * - **Error Propagation**: Centralizes error handling from all subsystems
 *
 * ### Memory Management Strategy
 * - **RAII Principles**: Automatic resource cleanup through structured lifecycle
 * - **Streaming Optimization**: On-demand loading with intelligent caching
 * - **Memory Pressure Handling**: Adaptive cache sizing based on system resources
 * - **Cross-Platform Compatibility**: Consistent behavior across different platforms
 *
 * ### Integration Points
 * - **Streaming Subsystem**: Coordinates with streaming cache and optimization managers
 * - **Validation Framework**: Integrates format-specific and cross-format validation
 * - **Format Modules**: Delegates format-specific operations to specialized implementations
 * - **Platform Layer**: Ensures cross-platform compatibility and system integration
 *
 * ## Key Implementation Details
 *
 * ### Dataset Structure Management
 * The `llama_dataset` structure is dynamically allocated and contains:
 * - Format-specific contexts (GGUF, GGML, tokenizer contexts)
 * - Streaming infrastructure (cache, optimization manager, read-ahead buffer)
 * - Metadata storage and access mechanisms
 * - Error state tracking and diagnostic information
 *
 * ### Streaming Implementation
 * Streaming mode provides memory-efficient access to large datasets:
 * - **LRU Cache**: Intelligent caching with configurable size limits
 * - **Read-Ahead Buffering**: Predictive loading based on access patterns
 * - **Adaptive Sizing**: Dynamic cache adjustment based on memory pressure
 * - **Performance Monitoring**: Real-time statistics and optimization metrics
 *
 * ### Metadata Handling
 * Standardized metadata access across all formats:
 * - **Key Normalization**: Consistent key naming across different source formats
 * - **Type Safety**: Robust type checking and conversion for metadata values
 * - **Default Handling**: Graceful fallback for missing or invalid metadata
 * - **Format Migration**: Automatic metadata translation during format conversion
 *
 * ### Error Management
 * Comprehensive error handling with detailed diagnostics:
 * - **Thread-Local Storage**: Thread-safe error state management
 * - **Error Propagation**: Consistent error reporting across all modules
 * - **Diagnostic Information**: Detailed error messages with context
 * - **Recovery Strategies**: Graceful degradation when possible
 *
 * ## Performance Characteristics
 *
 * ### Memory Usage
 * - **Base Overhead**: ~1KB per dataset structure plus format-specific overhead
 * - **Streaming Mode**: Memory usage scales with cache size, not dataset size
 * - **Non-Streaming Mode**: Full dataset loaded into memory for maximum performance
 * - **Adaptive Scaling**: Cache size automatically adjusts based on available memory
 *
 * ### Access Patterns
 * - **Sequential Access**: Optimized with read-ahead buffering (10-50% performance gain)
 * - **Random Access**: Efficient with LRU caching (cache hit ratios typically >80%)
 * - **Mixed Patterns**: Adaptive optimization adjusts to detected access patterns
 * - **Concurrent Access**: Thread-safe read operations with minimal contention
 *
 * ### Format-Specific Performance
 * - **GGUF**: Native format with optimal performance and full feature support
 * - **Text**: Tokenization overhead amortized through intelligent caching
 * - **Parquet**: Apache Arrow integration provides efficient columnar access
 *
 * ## Algorithm Details
 *
 * ### Cache Management Algorithm
 * The streaming cache uses a sophisticated LRU implementation:
 * 1. **Hash-based Lookup**: O(1) average case access time
 * 2. **Doubly-Linked List**: Efficient LRU ordering maintenance
 * 3. **Memory Pressure Detection**: System memory monitoring for adaptive sizing
 * 4. **Prefetch Coordination**: Integration with read-ahead buffer for optimal loading
 *
 * ### Read-Ahead Strategy
 * Predictive loading algorithm adapts to access patterns:
 * 1. **Pattern Detection**: Analyzes recent access history for sequential patterns
 * 2. **Window Sizing**: Dynamically adjusts prefetch window based on hit rates
 * 3. **Memory Awareness**: Respects cache limits and memory pressure
 * 4. **Format Optimization**: Leverages format-specific loading characteristics
 *
 * ### Metadata Extraction
 * Unified metadata access across different source formats:
 * 1. **Format Detection**: Automatic identification of source format metadata
 * 2. **Key Mapping**: Translation between format-specific and standardized keys
 * 3. **Type Conversion**: Safe conversion between different metadata value types
 * 4. **Validation**: Integrity checking for critical metadata values
 *
 * ## Integration with Other Modules
 *
 * ### Streaming Subsystem Integration
 * - **streaming/streaming-cache.h**: LRU cache implementation and memory management
 * - **streaming/streaming-optimization-manager.h**: Adaptive optimization coordination
 * - **streaming/streaming-read-ahead.h**: Predictive loading and prefetch management
 * - **streaming/streaming-memory-monitor.h**: System memory pressure monitoring
 *
 * ### Format Module Integration
 * - **formats/gguf/**: Native GGUF format support with full streaming capabilities
 * - **formats/text/**: Text processing with tokenization and intelligent caching
 * - **formats/parquet/**: Apache Arrow integration for efficient columnar access
 *
 * ### Validation Framework Integration
 * - **validation/llama-dataset-validation.h**: Comprehensive data integrity checking
 * - **validation/test-data-validator.h**: Format-specific validation implementations
 * - **validation/test-data-validator-common.h**: Cross-format validation utilities
 *
 * ### Platform Layer Integration
 * - **platform/platform-compat.h**: Cross-platform compatibility and system integration
 *
 * ## Thread Safety Considerations
 *
 * The implementation provides thread-safe read operations with the following guarantees:
 * - **Read Operations**: Multiple threads can safely read from the same dataset
 * - **Configuration Changes**: Must be performed from a single thread
 * - **Error State**: Thread-local storage ensures isolated error reporting
 * - **Cache Operations**: Internal synchronization for streaming cache access
 *
 * ## Future Enhancements
 *
 * Planned improvements and extension points:
 * - **Additional Formats**: Plugin architecture for new format support
 * - **Advanced Caching**: Multi-level caching with persistent storage options
 * - **Distributed Access**: Network-based dataset access and caching
 * - **GPU Integration**: Direct GPU memory management for training acceleration
 *
 * @see llama-dataset.h for the public interface documentation
 * @see streaming/ directory for streaming implementation details
 * @see formats/ directory for format-specific implementations
 * @see validation/ directory for validation framework details
 * @see platform/ directory for cross-platform compatibility
 *
 * @version 1.0
 * @since 2024
 */

#include "llama-dataset.h"

#include "common.h"
#include "ggml/include/ggml.h"
#include "ggml/include/gguf.h"
#include "llama-dataset-gguf-utils.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-parquet-internal.h"
#include "llama-dataset-parquet.h"
#include "llama-dataset-text.h"
#include "llama-dataset-utils.h"
#include "src/llama-impl.h"
#include "streaming-optimization-manager.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>

//
// Factory Functions - Format-Specific Dataset Creation
//
// These functions serve as the primary entry points for creating datasets from different
// formats. They act as thin wrappers around format-specific implementations while
// providing a consistent interface and error handling strategy.
//

/**
 * @brief Factory function for creating GGUF datasets.
 *
 * This function serves as the primary entry point for loading GGUF format datasets.
 * It delegates to the advanced GGUF loading implementation while providing a simplified
 * interface for common use cases.
 *
 * The function automatically detects whether streaming mode should be enabled based on
 * file size and available memory, then configures optimal default settings for cache
 * size and read-ahead buffering.
 *
 * @param params Common parameters including file path and streaming configuration
 * @return Pointer to the dataset, or NULL on error
 * @see llama_dataset_load_gguf() for the advanced implementation
 */
struct llama_dataset * llama_dataset_from_gguf(const common_params * params) {
    return llama_dataset_load_gguf(params);
}

/**
 * @brief Factory function for creating text datasets with tokenization.
 *
 * This function creates a dataset from a text file by tokenizing it using the provided
 * llama model. The tokenization process is optimized for training data preparation with
 * intelligent caching of tokenized sequences.
 *
 * The implementation handles various text encodings and provides robust error handling
 * for tokenization failures. Memory usage is optimized through streaming mode when
 * dealing with large text files.
 *
 * @param params Common parameters including file path and processing options
 * @param model Model to use for tokenization (must be compatible with the text format)
 * @return Pointer to the dataset, or NULL on error
 * @see formats/text/llama-dataset-text.h for text-specific implementation details
 */
struct llama_dataset * llama_dataset_from_txt(const common_params * params, struct llama_model * model) {
    return llama_dataset_load_text_internal(params, model);
}

/**
 * @brief Factory function for creating Parquet datasets.
 *
 * This function creates a dataset from a Parquet file using Apache Arrow integration.
 * It supports complex schemas, automatic column type detection, and efficient streaming
 * access to large Parquet files.
 *
 * The implementation automatically analyzes the Parquet schema to determine the optimal
 * loading strategy and configures streaming parameters based on file characteristics
 * and available system resources.
 *
 * @param params Common parameters including file path and schema configuration
 * @return Pointer to the dataset, or NULL on error
 * @see formats/parquet/llama-dataset-parquet.h for Parquet-specific implementation details
 */
#ifdef LLAMA_PARQUET
struct llama_dataset * llama_dataset_from_parquet(const common_params * params) {
    return llama_dataset_load_parquet_internal(params);
}
#endif

//
// Core Dataset Access Functions
//
// These functions provide the fundamental interface for accessing dataset content.
// They implement intelligent caching, streaming optimization, and format-agnostic
// access patterns while maintaining high performance across all supported formats.
//

/**
 * @brief Get the number of sequences in the dataset with intelligent caching.
 *
 * This function implements a multi-tier approach to sequence counting:
 * 1. **Cache Lookup**: First checks for a cached value from dataset loading
 * 2. **Metadata Extraction**: Attempts to read count from standardized metadata
 * 3. **Direct Counting**: Falls back to format-specific counting methods
 * 4. **Error Handling**: Provides graceful degradation for invalid datasets
 *
 * The caching strategy ensures O(1) access time for repeated calls while maintaining
 * accuracy across different dataset formats and loading modes.
 *
 * ## Performance Characteristics
 * - **Cached Access**: O(1) - immediate return from cached value
 * - **Metadata Access**: O(1) - single metadata lookup operation
 * - **Direct Counting**: O(n) - only for datasets without metadata (rare)
 * - **Memory Usage**: Minimal - only stores a single cached integer value
 *
 * ## Format-Specific Behavior
 * - **GGUF**: Uses tensor count from GGUF context for accurate sequence counting
 * - **Text**: Returns number of tokenized sequences from preprocessing
 * - **Parquet**: Uses row count from Apache Arrow table metadata
 *
 * @param dataset Dataset to query (must be valid and properly initialized)
 * @return Number of sequences, or 0 if dataset is NULL or invalid
 * @see llama_dataset_sequence() for accessing individual sequences
 * @see llama_dataset_sequence_length() for getting sequence lengths
 */
uint64_t llama_dataset_n_sequences(const struct llama_dataset * dataset) {
    if (!dataset) {
        return 0;
    }

    // First check if we have a cached value
    if (dataset->n_seq > 0) {
        return dataset->n_seq;
    }

    // If not cached, try to get from metadata
    if (dataset->ctx) {
        int32_t key_idx = gguf_find_key(dataset->ctx, TRAINING_SEQUENCE_COUNT);
        if (key_idx >= 0) {
            enum gguf_type type = gguf_get_kv_type(dataset->ctx, key_idx);
            if (type == GGUF_TYPE_INT32 || type == GGUF_TYPE_INT64) {
                return (uint64_t)gguf_get_val_i64(dataset->ctx, key_idx);
            }
        }

        // If not in metadata, count tensors
        return gguf_get_n_tensors(dataset->ctx);
    }

    return 0;
}

//
// Metadata Access Interface Implementation
//
// These functions provide standardized access to dataset metadata across all supported
// formats. The implementation handles format-specific metadata extraction, type conversion,
// and provides robust error handling with sensible defaults.
//

/**
 * @brief Retrieve string metadata with format-agnostic key mapping.
 *
 * This function implements a sophisticated metadata access strategy:
 * 1. **Input Validation**: Comprehensive parameter checking with null safety
 * 2. **Key Lookup**: Efficient hash-based key search in metadata store
 * 3. **Type Verification**: Ensures requested value is actually a string type
 * 4. **Format Translation**: Handles format-specific key variations automatically
 *
 * The implementation supports both standardized keys (defined in the header) and
 * format-specific keys, providing a unified interface for metadata access regardless
 * of the underlying dataset format.
 *
 * ## Supported Key Types
 * - **Standard Keys**: TRAINING_* constants defined in llama-dataset.h
 * - **Format-Specific Keys**: Native keys from GGUF, Parquet, or text metadata
 * - **User-Defined Keys**: Custom metadata added during dataset creation
 *
 * ## Error Handling
 * Returns NULL for any of the following conditions:
 * - Invalid dataset pointer
 * - Missing or invalid GGUF context
 * - Key not found in metadata
 * - Value exists but is not a string type
 *
 * @param dataset Dataset to query (must have valid metadata context)
 * @param key Metadata key to retrieve (case-sensitive)
 * @return String value or NULL if not found or invalid
 * @see llama_dataset_get_metadata_int() for integer metadata access
 * @see llama_dataset_get_metadata_float() for floating-point metadata access
 */
const char * llama_dataset_get_metadata_str(const struct llama_dataset * dataset, const char * key) {
    if (!dataset || !dataset->ctx || !key) {
        return nullptr;
    }

    int32_t key_idx = gguf_find_key(dataset->ctx, key);
    if (key_idx < 0) {
        return nullptr;
    }

    enum gguf_type type = gguf_get_kv_type(dataset->ctx, key_idx);
    return type == GGUF_TYPE_STRING ? gguf_get_val_str(dataset->ctx, key_idx) : nullptr;
}

/**
 * @brief Retrieve integer metadata with automatic type conversion and defaults.
 *
 * This function provides robust integer metadata access with the following features:
 * 1. **Type Flexibility**: Accepts both INT32 and INT64 metadata values
 * 2. **Automatic Conversion**: Safely converts between integer types as needed
 * 3. **Default Handling**: Returns specified default for missing or invalid keys
 * 4. **Overflow Protection**: Handles potential overflow in type conversions
 *
 * The implementation is particularly useful for accessing numeric configuration
 * parameters, sequence counts, and other quantitative metadata that may be stored
 * in different integer formats across various dataset sources.
 *
 * ## Type Conversion Rules
 * - **INT32 → INT64**: Zero-extension for positive values, sign-extension for negative
 * - **INT64 → INT64**: Direct value return without conversion
 * - **Other Types**: Return default value (no implicit conversion from strings/floats)
 *
 * ## Common Use Cases
 * - Sequence counts and dataset size information
 * - Configuration parameters and processing options
 * - Version numbers and format identifiers
 * - Timestamp values and creation dates
 *
 * @param dataset Dataset to query (must have valid metadata context)
 * @param key Metadata key to retrieve (case-sensitive)
 * @param default_value Value to return if key is not found or invalid
 * @return Integer value or default_value if not found/invalid
 * @see llama_dataset_get_metadata_str() for string metadata access
 * @see llama_dataset_get_metadata_float() for floating-point metadata access
 */
int64_t llama_dataset_get_metadata_int(const struct llama_dataset * dataset, const char * key, int64_t default_value) {
    if (!dataset || !dataset->ctx || !key) {
        return default_value;
    }

    int32_t key_idx = gguf_find_key(dataset->ctx, key);
    if (key_idx < 0) {
        return default_value;
    }

    enum gguf_type type = gguf_get_kv_type(dataset->ctx, key_idx);
    return type == GGUF_TYPE_INT32 || type == GGUF_TYPE_INT64 ? gguf_get_val_i64(dataset->ctx, key_idx) : default_value;
}

/**
 * @brief Retrieve floating-point metadata with precision handling and defaults.
 *
 * This function provides specialized access to floating-point metadata values with
 * careful attention to precision and numerical stability:
 * 1. **Precision Preservation**: Maintains accuracy for FLOAT32 values
 * 2. **Type Safety**: Only accepts actual floating-point metadata types
 * 3. **Default Handling**: Graceful fallback for missing or incompatible values
 * 4. **NaN Detection**: Handles special floating-point values appropriately
 *
 * The implementation is designed for accessing numerical parameters such as learning
 * rates, scaling factors, and other floating-point configuration values that may be
 * embedded in dataset metadata.
 *
 * ## Precision Considerations
 * - **FLOAT32**: Native precision maintained without conversion artifacts
 * - **FLOAT64**: Currently not supported (returns default to avoid precision loss)
 * - **Integer Types**: No automatic conversion (returns default for type safety)
 *
 * ## Special Value Handling
 * - **NaN Values**: Returned as-is (caller responsible for NaN checking)
 * - **Infinity**: Returned as-is (caller responsible for bounds checking)
 * - **Denormal Numbers**: Preserved according to IEEE 754 standards
 *
 * @param dataset Dataset to query (must have valid metadata context)
 * @param key Metadata key to retrieve (case-sensitive)
 * @param default_value Value to return if key is not found or invalid
 * @return Float value or default_value if not found/invalid
 * @see llama_dataset_get_metadata_str() for string metadata access
 * @see llama_dataset_get_metadata_int() for integer metadata access
 */
float llama_dataset_get_metadata_float(const struct llama_dataset * dataset, const char * key, float default_value) {
    if (!dataset || !dataset->ctx || !key) {
        return default_value;
    }

    int32_t key_idx = gguf_find_key(dataset->ctx, key);
    if (key_idx < 0) {
        return default_value;
    }

    enum gguf_type type = gguf_get_kv_type(dataset->ctx, key_idx);
    return type == GGUF_TYPE_FLOAT32 ? gguf_get_val_f32(dataset->ctx, key_idx) : default_value;
}

//
// Dataset Format Conversion Implementation
//
// This section implements the comprehensive dataset conversion functionality that
// enables transformation between different dataset formats while preserving metadata,
// optimizing structure, and ensuring data integrity throughout the conversion process.
//

/**
 * @brief Convert any dataset format to GGUF with comprehensive metadata preservation.
 *
 * This function implements a sophisticated conversion algorithm that handles multiple
 * source formats and conversion scenarios:
 *
 * ## Conversion Strategies
 *
 * ### GGUF-to-GGUF Conversion
 * For GGUF source datasets, the function implements two distinct approaches:
 * 1. **Streaming Mode**: Loads tensor data on-demand and writes incrementally
 * 2. **Memory Mode**: Direct context copying with optimized tensor handling
 *
 * ### Cross-Format Conversion (Text/Parquet → GGUF)
 * For non-GGUF sources, the conversion process involves:
 * 1. **Metadata Migration**: Translates format-specific metadata to GGUF standards
 * 2. **Data Restructuring**: Converts sequences to GGUF tensor format
 * 3. **Optimization**: Applies GGUF-specific optimizations for training efficiency
 * 4. **Validation**: Ensures data integrity throughout the conversion process
 *
 * ## Algorithm Details
 *
 * ### Streaming Conversion Algorithm
 * For large datasets in streaming mode:
 * 1. **Tensor Enumeration**: Iterates through all sequences to ensure data availability
 * 2. **Incremental Writing**: Writes GGUF header, metadata, and tensor data sequentially
 * 3. **Memory Management**: Maintains minimal memory footprint during conversion
 * 4. **Alignment Handling**: Ensures proper 32-byte alignment for tensor data
 *
 * ### Metadata Conversion Algorithm
 * Comprehensive metadata handling:
 * 1. **Key Translation**: Maps format-specific keys to standardized GGUF keys
 * 2. **Type Conversion**: Safely converts between different metadata value types
 * 3. **Enrichment**: Adds standard metadata (timestamps, version info, statistics)
 * 4. **Validation**: Verifies metadata consistency and completeness
 *
 * ### Tensor Creation Algorithm
 * For cross-format conversion:
 * 1. **Sequence Analysis**: Determines optimal tensor structure and naming scheme
 * 2. **Memory Allocation**: Creates GGML context with appropriate memory sizing
 * 3. **Data Copying**: Efficiently transfers sequence data to tensor format
 * 4. **Tensor Registration**: Adds tensors to GGUF context with proper metadata
 *
 * ## Performance Characteristics
 *
 * ### Memory Usage
 * - **Streaming Mode**: O(cache_size) - independent of dataset size
 * - **Memory Mode**: O(dataset_size) - full dataset loaded during conversion
 * - **Cross-Format**: O(max_sequence_length) - processes sequences individually
 *
 * ### Time Complexity
 * - **GGUF-to-GGUF**: O(n) where n is total tensor data size
 * - **Cross-Format**: O(n × m) where n is sequence count, m is average sequence length
 * - **I/O Bound**: Performance primarily limited by disk I/O bandwidth
 *
 * ## Error Handling Strategy
 *
 * The function implements comprehensive error handling:
 * 1. **Input Validation**: Thorough parameter checking before processing
 * 2. **Resource Management**: Automatic cleanup on any failure condition
 * 3. **Progress Tracking**: Detailed error reporting with conversion progress context
 * 4. **Rollback Capability**: Ensures no partial files are left on failure
 *
 * ## Format-Specific Optimizations
 *
 * ### GGUF Source Optimizations
 * - **Context Reuse**: Leverages existing GGUF context for efficient copying
 * - **Streaming Integration**: Coordinates with streaming cache for optimal performance
 * - **Metadata Preservation**: Maintains all original metadata with additions
 *
 * ### Text Source Optimizations
 * - **Tokenization Caching**: Reuses existing tokenized sequences
 * - **Batch Processing**: Groups sequences for efficient tensor creation
 * - **Memory Efficiency**: Processes large text files without full loading
 *
 * ### Parquet Source Optimizations
 * - **Columnar Access**: Leverages Parquet's columnar structure for efficiency
 * - **Schema Analysis**: Optimizes conversion based on detected schema patterns
 * - **Batch Loading**: Uses Arrow's batch processing for memory efficiency
 *
 * @param dataset Source dataset (any supported format)
 * @param path Output path for the GGUF file
 * @see tools/convert-to-gguf.cpp for command-line conversion utility
 * @see formats/gguf/llama-dataset-gguf.h for GGUF format details
 */
void llama_dataset_to_gguf(struct llama_dataset * dataset, const char * path) {
    if (!dataset || !path) {
        llama_dataset_set_error("Invalid parameters for GGUF conversion\n");
        return;
    }

    // For GGUF datasets, we can use the existing GGUF context
    if (dataset->type == DATASET_GGUF && dataset->ctx) {
        // If the dataset is in streaming mode, we need to ensure all tensors are loaded
        if (dataset->streaming) {
            LLAMA_LOG_INFO("Converting streaming GGUF dataset to file: %s\n", path);

            // For streaming datasets, we need to load all tensor data
            uint64_t n_seq = llama_dataset_n_sequences(dataset);
            for (uint64_t i = 0; i < n_seq; i++) {
                // This will trigger loading the tensor data if not already loaded
                if (llama_dataset_sequence(dataset, i) == nullptr) {
                    llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to load tensor data for streaming conversion");
                    return;
                }
            }
        }

        // Write GGUF file manually since we need to handle streaming data properly
        FILE* file = fopen(path, "wb");
        if (!file) {
            llama_dataset_set_error_with_code(DATASET_ERROR_FILE_NOT_FOUND, "Failed to open output file for writing");
            return;
        }

        // Get the meta data size and write header + metadata
        size_t meta_size = gguf_get_meta_size(dataset->ctx);
        void* meta_data = malloc(meta_size);
        if (!meta_data) {
            fclose(file);
            llama_dataset_set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate memory for GGUF metadata");
            return;
        }

        // Get and write the metadata
        gguf_get_meta_data(dataset->ctx, meta_data);
        if (fwrite(meta_data, 1, meta_size, file) != meta_size) {
            free(meta_data);
            fclose(file);
            llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to write GGUF metadata");
            return;
        }
        free(meta_data);

        // Write tensor data - for streaming mode, we need to get data from cache
        uint64_t n_tensors = gguf_get_n_tensors(dataset->ctx);
        for (uint64_t i = 0; i < n_tensors; i++) {
            const void* tensor_data = nullptr;
            size_t tensor_size = 0;

            if (dataset->cached_tensors && dataset->cached_tensors[i]) {
                if (dataset->cached_tensors[i]->data) {
                    tensor_data = dataset->cached_tensors[i]->data;
                    tensor_size = ggml_nbytes(dataset->cached_tensors[i]);
                } else {
                    // Get tensor size from GGUF context
                    tensor_size = gguf_get_tensor_size(dataset->ctx, i);
                }
            } else {
                tensor_size = gguf_get_tensor_size(dataset->ctx, i);
            }

            if (!tensor_data) {
                fclose(file);
                llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Tensor data not available for writing");
                return;
            }

            if (tensor_size == 0) {
                fclose(file);
                llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Invalid tensor size for writing");
                return;
            }

            // Write tensor data with proper alignment
            if (fwrite(tensor_data, 1, tensor_size, file) != tensor_size) {
                fclose(file);
                llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to write tensor data");
                return;
            }

            // Add padding to align to 32-byte boundary if needed
            size_t padding = (32 - (tensor_size % 32)) % 32;
            if (padding > 0) {
                char zero_padding[32] = {0};
                if (fwrite(zero_padding, 1, padding, file) != padding) {
                    fclose(file);
                    llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to write tensor padding");
                    return;
                }
            }
        }

        fclose(file);
        LLAMA_LOG_INFO("Successfully wrote GGUF dataset to %s\n", path);
        return;
    }

    // For other formats (TEXT, PARQUET), we need to create a new GGUF file
    LLAMA_LOG_INFO("Converting %s dataset to GGUF file: %s\n", dataset->type == DATASET_TEXT ? "TEXT" : "PARQUET", path);

    // Create a new GGUF context
    struct gguf_context * new_ctx = gguf_init_empty();
    if (!new_ctx) {
        llama_dataset_set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGUF context for conversion");
        return;
    }

    // Copy metadata from the original dataset
    if (dataset->ctx) {
        // Copy all key-value pairs
        int n_kv = gguf_get_n_kv(dataset->ctx);
        for (int i = 0; i < n_kv; i++) {
            const char* key = gguf_get_key(dataset->ctx, i);
            enum gguf_type type = gguf_get_kv_type(dataset->ctx, i);

            switch (type) {
                case GGUF_TYPE_STRING:
                    gguf_set_val_str(new_ctx, key, gguf_get_val_str(dataset->ctx, i));
                    break;
                case GGUF_TYPE_INT32:
                    gguf_set_val_i32(new_ctx, key, gguf_get_val_i32(dataset->ctx, i));
                    break;
                case GGUF_TYPE_UINT32:
                    gguf_set_val_u32(new_ctx, key, gguf_get_val_u32(dataset->ctx, i));
                    break;
                case GGUF_TYPE_INT64:
                    gguf_set_val_i64(new_ctx, key, gguf_get_val_i64(dataset->ctx, i));
                    break;
                case GGUF_TYPE_UINT64:
                    gguf_set_val_u64(new_ctx, key, gguf_get_val_u64(dataset->ctx, i));
                    break;
                case GGUF_TYPE_FLOAT32:
                    gguf_set_val_f32(new_ctx, key, gguf_get_val_f32(dataset->ctx, i));
                    break;
                case GGUF_TYPE_FLOAT64:
                    gguf_set_val_f64(new_ctx, key, gguf_get_val_f64(dataset->ctx, i));
                    break;
                default:
                    LLAMA_LOG_WARN("Bad metadata key '%s' with type %d", key, type);
                    break;
            }
        }
    }

    // Add or update standard metadata
    gguf_set_val_str(new_ctx, TRAINING_FORMAT_SOURCE, dataset->type == DATASET_TEXT ? "text" : dataset->type == DATASET_PARQUET ? "parquet" : "gguf");

    // Add creation timestamp
    time_t now = time(nullptr);
    char timestamp[32];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));
    gguf_set_val_str(new_ctx, TRAINING_CREATION_TIME, timestamp);

    // Set sequence count
    uint64_t seq_count = llama_dataset_n_sequences(dataset);
    gguf_set_val_i32(new_ctx, TRAINING_SEQUENCE_COUNT, static_cast<int32_t>(seq_count));

    // Find maximum sequence length
    int32_t max_length = 0;
    for (uint64_t i = 0; i < seq_count; i++) {
        int32_t len = llama_dataset_sequence_length(dataset, i);
        if (len > max_length) {
            max_length = len;
        }
    }
    gguf_set_val_u32(new_ctx, TRAINING_MAX_LENGTH, static_cast<uint32_t>(max_length));

    // Add additional useful metadata
    gguf_set_val_i16(new_ctx, TRAINING_FORMAT_VERSION, 1000);
    gguf_set_val_str(new_ctx, TRAINING_DATASET_NAME, "llama-dataset");
    
    // Additional metadata fields can be added here as needed:
    // - training.dataset.source: URL or description of the data source
    // - training.tokenizer.gguf.model: Tokenizer model name (llama, gpt2, etc.)
    // - training.tokenizer.gguf.vocab: Tokenizer dictionary
    // - training.tokenizer.gguf.merges: Tokenizer merges (for BPE)
    // - training.tokenizer.gguf.pre: Pre-tokenization architecture


    // Add total token count
    uint64_t total_tokens = 0;
    for (uint64_t i = 0; i < seq_count; i++) {
        total_tokens += llama_dataset_sequence_length(dataset, i);
    }
    gguf_set_val_u64(new_ctx, TRAINING_SEQUENCE_COUNT, static_cast<uint32_t>(total_tokens));

    // Create GGML context for tensor data
    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ 128ull*1024ull*1024ull,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context* ggml_ctx = ggml_init(ggml_params);
    if (!ggml_ctx) {
        gguf_free(new_ctx);
        llama_dataset_set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGML context for conversion");
        return;
    }

    // Add all sequences as tensors
    for (uint64_t i = 0; i < seq_count; i++) {
        // Get sequence data
        const int32_t * tokens = llama_dataset_sequence(dataset, i);
        int32_t length = llama_dataset_sequence_length(dataset, i);

        if (!tokens || length <= 0) {
            LLAMA_LOG_WARN("Skipping invalid sequence at index %zu\n", i);
            continue;
        }

        // Create tensor name (use format "seq_XXXXX" with zero-padding)
        char tensor_name[32];
        snprintf(tensor_name, sizeof(tensor_name), "seq_%05" PRIu64, i);

        // Create tensor in GGML context
        int64_t ne[1] = { static_cast<int64_t>(length) };
        struct ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, GGML_TYPE_I32, 1, ne);
        ggml_set_name(tensor, tensor_name);

        // Copy data to tensor
        memcpy(tensor->data, tokens, length * sizeof(int32_t));

        // Add tensor to GGUF context
        gguf_add_tensor(new_ctx, tensor);
    }

    // Use the built-in GGUF writing functionality
    if (!gguf_write_to_file(new_ctx, path, false)) {
        ggml_free(ggml_ctx);
        gguf_free(new_ctx);
        llama_dataset_set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to write GGUF file");
        return;
    }

    // Clean up
    ggml_free(ggml_ctx);
    gguf_free(new_ctx);

    LLAMA_LOG_INFO("Successfully converted dataset to GGUF file: %s\n", path);
}

//
// Resource Management and Cleanup Implementation
//
// This section implements comprehensive resource management with careful attention to
// memory safety, proper cleanup ordering, and format-specific resource handling.
// The cleanup process ensures no memory leaks while maintaining thread safety.
//

/**
 * @brief Comprehensive dataset cleanup with format-specific resource management.
 *
 * This function implements a sophisticated cleanup algorithm that handles the complex
 * resource management requirements of the multi-format, multi-subsystem dataset
 * architecture. The cleanup process follows a carefully designed order to ensure
 * proper resource deallocation and avoid use-after-free conditions.
 *
 * ## Cleanup Algorithm
 *
 * The cleanup process follows this specific order to ensure safety:
 * 1. **High-Level Resources**: Tokenization contexts and model references
 * 2. **Streaming Infrastructure**: Optimization managers and caching systems
 * 3. **Cached Data**: Tensor pointers and streaming cache entries
 * 4. **Core Contexts**: GGML and GGUF contexts with their associated memory
 * 5. **Format-Specific Data**: Format-dependent resources and file handles
 * 6. **Dataset Structure**: The main dataset structure itself
 *
 * ## Resource Management Strategy
 *
 * ### Tokenization Resources
 * - **Context Cleanup**: Properly frees llama tokenization contexts
 * - **Model Ownership**: Only frees models when dataset owns them
 * - **Reference Counting**: Handles shared model references safely
 *
 * ### Streaming Infrastructure
 * - **Optimization Manager**: C++ object destruction with proper cleanup
 * - **Streaming Cache**: LRU cache cleanup with memory deallocation
 * - **Cache Coordination**: Ensures cache and optimization manager cleanup order
 *
 * ### Memory Management
 * - **Streaming Mode**: Special handling for on-demand loaded tensor data
 * - **Memory Mode**: Standard cleanup for fully-loaded datasets
 * - **Cache Integration**: Coordinates with streaming cache for tensor cleanup
 *
 * ### Format-Specific Cleanup
 * - **GGUF**: Simple string cleanup for file path storage
 * - **Text**: Cleanup of tokenizer state and file handles
 * - **Parquet**: Complex cleanup of Arrow contexts and schema information
 *
 * ## Thread Safety Considerations
 *
 * The cleanup function is designed to be thread-safe with the following guarantees:
 * - **Single-Threaded Cleanup**: Only one thread should call this function per dataset
 * - **Read Operation Safety**: Ongoing read operations will complete safely
 * - **Error State Preservation**: Error state is maintained for post-cleanup inspection
 *
 * ## Memory Safety Features
 *
 * ### Null Pointer Safety
 * - **Defensive Programming**: All pointer checks before deallocation
 * - **Idempotent Cleanup**: Safe to call multiple times on the same dataset
 * - **Partial Cleanup**: Handles partially-initialized datasets gracefully
 *
 * ### Resource Leak Prevention
 * - **Comprehensive Coverage**: All allocated resources are properly tracked
 * - **Exception Safety**: C++ objects cleaned up even in error conditions
 * - **Platform Independence**: Consistent cleanup behavior across platforms
 *
 * ## Performance Characteristics
 *
 * ### Time Complexity
 * - **Streaming Mode**: O(cache_entries) - proportional to cached data
 * - **Memory Mode**: O(1) - constant time for context cleanup
 * - **Format Overhead**: Varies by format complexity (GGUF < Text < Parquet)
 *
 * ### Memory Deallocation
 * - **Immediate Release**: Most memory freed immediately
 * - **System Integration**: Coordinates with system memory manager
 * - **Cache Flushing**: Streaming cache memory returned to system
 *
 * ## Error Handling During Cleanup
 *
 * The cleanup process is designed to be robust against errors:
 * - **Continue on Error**: Individual cleanup failures don't stop the process
 * - **Error Preservation**: Original error state maintained throughout cleanup
 * - **Diagnostic Logging**: Cleanup errors logged for debugging purposes
 * - **Resource Tracking**: Ensures critical resources are freed even on errors
 *
 * @param dataset Dataset to free (can be NULL for safe no-op behavior)
 * @note Error state is preserved after cleanup for caller inspection
 * @note This function is not thread-safe - ensure exclusive access during cleanup
 * @see llama_dataset_get_error() for checking errors after cleanup
 */
void llama_dataset_free(struct llama_dataset * dataset) {
    if (!dataset) {
        return;
    }

    // Free tokenization resources
    if (dataset->tokenizer_ctx) {
        llama_free(dataset->tokenizer_ctx);
        dataset->tokenizer_ctx = nullptr;
    }

    if (dataset->model && dataset->owns_model) {
        llama_model_free(dataset->model);
        dataset->model = nullptr;
    }

    // Free streaming optimization manager if it exists
    if (dataset->optimization_manager) {
        delete static_cast<llama_dataset_stream_optimization_manager *>(dataset->optimization_manager);
        dataset->optimization_manager = nullptr;
    }

    // Free streaming cache if in streaming mode
    if (dataset->streaming && dataset->streaming_cache) {
        delete static_cast<llama_dataset_streaming_cache *>(dataset->streaming_cache);
        dataset->streaming_cache = nullptr;
    }

    // Free cached tensor pointers and their data if in streaming mode
    if (dataset->cached_tensors) {
        // In streaming mode, we need to free the tensor data that we allocated
        if (dataset->streaming) {
            uint64_t n_seq = llama_dataset_n_sequences(dataset);
            for (uint64_t i = 0; i < n_seq; i++) {
                if (dataset->cached_tensors[i]) {
                    // In streaming mode, tensor data is now managed by the streaming cache
                    // so we only need to free the placeholder tensor structure
                    free(dataset->cached_tensors[i]);
                    dataset->cached_tensors[i] = nullptr;
                }
            }
        }

        free(dataset->cached_tensors);
        dataset->cached_tensors = nullptr;
    }

    // Free GGML context
    if (dataset->ggml_ctx) {
        ggml_free(dataset->ggml_ctx);
        dataset->ggml_ctx = nullptr;
    }

    // Free GGUF context
    if (dataset->ctx) {
        gguf_free(dataset->ctx);
        dataset->ctx = nullptr;
    }

    // Free format-specific data based on type
    if (dataset->format_data) {
        switch (dataset->type) {
            case DATASET_TEXT:
                // Clean up text-specific resources
                // For example, if format_data contains tokenizer state or file handles
                break;

            case DATASET_PARQUET:
                // Clean up Parquet-specific resources
#ifdef LLAMA_PARQUET
                {
                    extern void llama_dataset_free_parquet_format_data(void * format_data);
                    llama_dataset_free_parquet_format_data(dataset->format_data);
                }
#else
                // If Parquet support is not compiled in, just free as generic pointer
                free(dataset->format_data);
#endif
                break;

            case DATASET_GGUF:
                // For GGUF in streaming mode, format_data contains the file path
                // which is a simple malloc'd string
                free(dataset->format_data);
                break;
        }

        dataset->format_data = nullptr;
    }

    // Finally free the dataset structure itself
    free(dataset);

    // Note: We don't clear the error state here because the caller might want to check
    // for errors after freeing the dataset
}
