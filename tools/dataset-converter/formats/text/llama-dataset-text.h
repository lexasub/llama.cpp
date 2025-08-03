#pragma once

/**
 * @file llama-dataset-text.h
 * @brief Text format dataset loader with advanced tokenization and streaming capabilities.
 *
 * This module provides comprehensive support for loading and processing text datasets within
 * the llama.cpp dataset converter framework. It handles text file parsing, tokenization using
 * llama models, and conversion to the standardized dataset format with full streaming support.
 *
 * ## Text Processing Pipeline
 *
 * The text processing pipeline consists of several stages:
 * 1. **File Reading**: Efficient line-by-line or chunk-based file reading
 * 2. **Text Preprocessing**: Optional text cleaning and normalization
 * 3. **Tokenization**: Conversion to tokens using the specified llama model
 * 4. **Sequence Formation**: Organization of tokens into training sequences
 * 5. **Caching**: Intelligent caching of tokenized sequences for performance
 * 6. **Streaming**: On-demand loading and processing for memory efficiency
 *
 * ## Key Features
 *
 * - **Flexible Tokenization**: Support for all llama model tokenizers
 * - **Streaming Mode**: Memory-efficient processing of large text files
 * - **Intelligent Caching**: LRU caching of tokenized sequences with adaptive sizing
 * - **Line-based Processing**: Efficient line-by-line tokenization with batching
 * - **Error Recovery**: Robust handling of encoding issues and malformed text
 * - **Performance Optimization**: Optimized tokenization pipeline with prefetching
 * - **Memory Management**: Automatic memory management with configurable limits
 * - **Progress Monitoring**: Built-in progress tracking for large file processing
 *
 * ## Supported Text Formats
 *
 * The module supports various text file formats and encodings:
 * - **Plain Text**: UTF-8 encoded text files (.txt, .text)
 * - **Line-delimited**: Each line represents a separate training sequence
 * - **Paragraph Mode**: Blank lines separate training sequences
 * - **Custom Delimiters**: Configurable sequence separation patterns
 * - **Large Files**: Efficient processing of multi-gigabyte text files
 *
 * ## Tokenization Process
 *
 * The tokenization process is optimized for training data preparation:
 * 
 * ### Standard Tokenization
 * ```c
 * // Load text dataset with default tokenization
 * struct llama_dataset* dataset = llama_dataset_from_txt(params, model);
 * 
 * // Access tokenized sequences
 * uint64_t count = llama_dataset_n_sequences(dataset);
 * const int32_t* tokens = llama_dataset_sequence(dataset, 0);
 * ```
 *
 * ### Advanced Configuration
 * ```c
 * // Configure streaming and caching
 * llama_dataset_set_streaming_cache_size(dataset, 256 * 1024 * 1024); // 256MB cache
 * llama_dataset_set_streaming_read_ahead(dataset, true, 20);           // Prefetch 20 sequences
 * llama_dataset_optimize_text_sequence_cache(dataset);                 // Optimize cache layout
 * ```
 *
 * ## Performance Considerations
 *
 * ### Memory Usage
 * - **Streaming Mode**: Processes text on-demand, minimal memory footprint
 * - **Cache Sizing**: Configurable cache size based on available memory
 * - **Batch Processing**: Efficient batching of tokenization operations
 * - **Memory Monitoring**: Automatic memory pressure detection and adaptation
 *
 * ### Processing Speed
 * - **Parallel Tokenization**: Multi-threaded tokenization for large files
 * - **Read-ahead Buffering**: Predictive loading of upcoming text segments
 * - **Optimized I/O**: Efficient file reading with configurable buffer sizes
 * - **Cache Optimization**: LRU caching with access pattern analysis
 *
 * ### Scalability
 * - **Large File Support**: Handles files larger than available memory
 * - **Progressive Loading**: Incremental processing with progress reporting
 * - **Adaptive Algorithms**: Dynamic optimization based on file characteristics
 * - **Resource Management**: Automatic resource cleanup and memory reclamation
 *
 * ## Integration with Core Framework
 *
 * This module integrates seamlessly with the core dataset framework:
 * - **Streaming Integration**: Full compatibility with streaming/streaming-cache.h
 * - **Validation Support**: Integration with validation/llama-dataset-validation.h
 * - **Metadata Management**: Automatic metadata extraction and standardization
 * - **Error Handling**: Consistent error reporting through core error interface
 * - **Performance Monitoring**: Integration with streaming performance metrics
 *
 * ## Error Handling and Recovery
 *
 * The module provides robust error handling for common text processing issues:
 * - **Encoding Errors**: Graceful handling of invalid UTF-8 sequences
 * - **File Access Issues**: Comprehensive file I/O error reporting
 * - **Tokenization Failures**: Recovery from tokenization errors with diagnostics
 * - **Memory Pressure**: Automatic cache reduction under memory constraints
 * - **Model Compatibility**: Validation of model-text compatibility
 *
 * ## Thread Safety
 *
 * The text processing module is designed for thread-safe operation:
 * - **Read Operations**: Thread-safe access to tokenized sequences
 * - **Cache Management**: Thread-safe cache operations with fine-grained locking
 * - **Tokenization**: Thread-safe tokenization using model contexts
 * - **Progress Reporting**: Thread-safe progress updates and statistics
 *
 * ## Usage Examples
 *
 * ### Basic Text Loading
 * ```c
 * // Load a simple text file
 * common_params params = {0};
 * params.input_file = "training_data.txt";
 * params.streaming = false;
 * 
 * struct llama_dataset* dataset = llama_dataset_from_txt(&params, model);
 * if (!dataset) {
 *     fprintf(stderr, "Error: %s\n", llama_dataset_get_error_message());
 *     return -1;
 * }
 * ```
 *
 * ### Streaming Large Files
 * ```c
 * // Configure for large file streaming
 * common_params params = {0};
 * params.input_file = "large_corpus.txt";
 * params.streaming = true;
 * params.cache_size = 512 * 1024 * 1024; // 512MB cache
 * 
 * struct llama_dataset* dataset = llama_dataset_from_txt(&params, model);
 * 
 * // Enable optimizations
 * llama_dataset_set_adaptive_cache_sizing(dataset, true);
 * llama_dataset_set_streaming_read_ahead(dataset, true, 50);
 * ```
 *
 * ### Performance Monitoring
 * ```c
 * // Monitor tokenization performance
 * double hit_ratio;
 * size_t memory_usage, entry_count;
 * 
 * if (llama_dataset_get_streaming_stats(dataset, &hit_ratio, &memory_usage, &entry_count)) {
 *     printf("Cache hit ratio: %.2f%%\n", hit_ratio * 100.0);
 *     printf("Memory usage: %zu MB\n", memory_usage / (1024 * 1024));
 *     printf("Cached sequences: %zu\n", entry_count);
 * }
 * ```
 *
 * @see core/llama-dataset.h for the main dataset interface
 * @see streaming/streaming-cache.h for streaming implementation details
 * @see validation/llama-dataset-validation.h for text validation capabilities
 * @see tools/convert-to-gguf.cpp for text-to-GGUF conversion utilities
 *
 * @version 1.0
 * @since 2024
 */

#include "llama-dataset.h"

// Forward declarations
struct llama_model;
typedef int32_t llama_token;

/**
 * @brief Load a dataset from a text file with comprehensive tokenization support.
 *
 * This function provides the internal implementation for loading text datasets with
 * advanced tokenization capabilities. It processes text files line-by-line or in
 * configurable chunks, applies the specified tokenization model, and creates a
 * dataset structure with full streaming and caching support.
 *
 * ## Processing Pipeline
 * 1. **File Validation**: Checks file accessibility, encoding, and format
 * 2. **Text Preprocessing**: Optional text cleaning and normalization
 * 3. **Tokenization**: Converts text to tokens using the provided llama model
 * 4. **Sequence Organization**: Groups tokens into training sequences
 * 5. **Cache Population**: Populates streaming cache with initial sequences
 * 6. **Metadata Extraction**: Generates dataset metadata and statistics
 *
 * ## Streaming Behavior
 * - **Enabled**: Text is tokenized on-demand with intelligent caching
 * - **Disabled**: Entire file is tokenized and loaded into memory
 * - **Adaptive**: Automatically switches based on file size and available memory
 *
 * ## Performance Optimizations
 * - **Batch Tokenization**: Processes multiple lines in batches for efficiency
 * - **Parallel Processing**: Multi-threaded tokenization for large files
 * - **Memory Mapping**: Uses memory-mapped I/O for very large files
 * - **Progressive Loading**: Provides progress feedback for long operations
 *
 * ## Error Handling
 * The function handles various error conditions gracefully:
 * - Invalid file paths or inaccessible files
 * - Unsupported text encodings or corrupted files
 * - Tokenization failures or model incompatibilities
 * - Memory allocation failures or resource constraints
 * - I/O errors during file processing
 *
 * @param params Common parameters including:
 *               - input_file: Path to the text file to process
 *               - streaming: Whether to enable streaming mode
 *               - cache_size: Size of the tokenization cache
 *               - max_sequence_length: Maximum tokens per sequence
 *               - batch_size: Number of lines to process in each batch
 * @param model Llama model to use for tokenization. Must be properly initialized
 *              and compatible with the target text format. The model's tokenizer
 *              will be used for all text-to-token conversions.
 * 
 * @return Pointer to the loaded dataset with tokenized sequences, or NULL on error.
 *         On error, detailed error information is available via llama_dataset_get_error().
 *         The returned dataset must be freed with llama_dataset_free().
 *
 * @note This function is thread-safe for concurrent access with different parameters,
 *       but the same model should not be used simultaneously from multiple threads.
 *
 * @see llama_dataset_from_txt() for the public interface
 * @see llama_dataset_tokenize_line() for single-line tokenization
 * @see llama_dataset_optimize_text_sequence_cache() for cache optimization
 *
 * @warning The model parameter must remain valid for the lifetime of the dataset
 *          when streaming mode is enabled, as tokenization occurs on-demand.
 */
struct llama_dataset * llama_dataset_load_text_internal(const common_params * params, struct llama_model * model);

/**
 * @brief Tokenize a single line of text with comprehensive error handling.
 *
 * This function provides low-level tokenization of individual text lines using
 * the specified llama model. It's optimized for batch processing and streaming
 * scenarios where lines need to be tokenized individually with precise control
 * over the tokenization process.
 *
 * ## Tokenization Process
 * 1. **Input Validation**: Validates input parameters and text encoding
 * 2. **Text Preprocessing**: Applies model-specific text preprocessing
 * 3. **Token Generation**: Converts text to tokens using the model's tokenizer
 * 4. **Buffer Management**: Safely writes tokens to the provided buffer
 * 5. **Error Detection**: Detects and reports tokenization errors
 *
 * ## Performance Characteristics
 * - **Optimized Path**: Uses the most efficient tokenization path for the model
 * - **Memory Efficient**: Minimal memory allocation during tokenization
 * - **Cache Friendly**: Designed to work efficiently with tokenization caches
 * - **Batch Compatible**: Optimized for use in batch tokenization scenarios
 *
 * ## Error Conditions
 * The function returns negative values for various error conditions:
 * - **-1**: Invalid model or model not properly initialized
 * - **-2**: Invalid input parameters (NULL pointers, invalid lengths)
 * - **-3**: Text encoding error or unsupported characters
 * - **-4**: Tokenization buffer too small for the result
 * - **-5**: Model tokenization failure or internal error
 *
 * ## Usage Patterns
 *
 * ### Single Line Processing
 * ```c
 * llama_token tokens[1024];
 * int32_t n_tokens = llama_dataset_tokenize_line(model, line, strlen(line), tokens, 1024);
 * if (n_tokens < 0) {
 *     fprintf(stderr, "Tokenization failed with error code: %d\n", n_tokens);
 * }
 * ```
 *
 * ### Batch Processing
 * ```c
 * for (int i = 0; i < n_lines; i++) {
 *     int32_t n_tokens = llama_dataset_tokenize_line(model, lines[i], line_lengths[i], 
 *                                                     token_buffers[i], max_tokens);
 *     if (n_tokens >= 0) {
 *         process_tokens(token_buffers[i], n_tokens);
 *     }
 * }
 * ```
 *
 * @param model Llama model to use for tokenization. Must be properly initialized
 *              with a compatible tokenizer. The model's context will be used
 *              for the tokenization process.
 * @param line Pointer to the text line to tokenize. Must be valid UTF-8 encoded
 *             text. The line does not need to be null-terminated if line_len
 *             is specified correctly.
 * @param line_len Length of the text line in bytes. Must be accurate and not
 *                 exceed the actual length of the text buffer. Use strlen()
 *                 for null-terminated strings.
 * @param tokens Buffer to store the resulting tokens. Must be large enough to
 *               hold the maximum expected number of tokens. The buffer will be
 *               filled with token IDs in the model's token space.
 * @param n_tokens_max Maximum number of tokens that can be stored in the buffer.
 *                     This prevents buffer overflows and should match the actual
 *                     size of the tokens array.
 *
 * @return Number of tokens generated (>= 0) on success, or negative error code on failure.
 *         The actual tokens are written to the provided buffer up to the returned count.
 *         A return value of 0 indicates an empty line or line with only whitespace.
 *
 * @note This function is thread-safe when used with different model instances,
 *       but the same model should not be used concurrently from multiple threads.
 *
 * @see llama_dataset_load_text_internal() for full text file processing
 * @see llama_dataset_optimize_text_sequence_cache() for cache optimization
 *
 * @warning The tokens buffer must be large enough to hold all resulting tokens.
 *          Insufficient buffer size will result in tokenization failure.
 */
int32_t llama_dataset_tokenize_line(struct llama_model * model, const char * line, int32_t line_len, llama_token * tokens, int32_t n_tokens_max);

/**
 * @brief Optimize text sequence cache layout and performance characteristics.
 *
 * This function performs comprehensive optimization of the text sequence cache
 * to improve access patterns, memory locality, and overall performance. It
 * analyzes current usage patterns and reorganizes the cache structure for
 * optimal efficiency based on the specific characteristics of the text dataset.
 *
 * ## Optimization Strategies
 *
 * ### Cache Layout Optimization
 * - **Memory Locality**: Reorganizes cache entries for better spatial locality
 * - **Access Pattern Analysis**: Analyzes recent access patterns to optimize layout
 * - **Prefetch Optimization**: Adjusts prefetch strategies based on usage patterns
 * - **Memory Alignment**: Ensures optimal memory alignment for cache entries
 *
 * ### Performance Tuning
 * - **LRU Optimization**: Fine-tunes LRU eviction policies for text access patterns
 * - **Batch Loading**: Optimizes batch loading of related sequences
 * - **Compression**: Applies sequence compression where beneficial
 * - **Index Optimization**: Optimizes sequence indexing for faster lookups
 *
 * ### Adaptive Configuration
 * - **Dynamic Sizing**: Adjusts cache size based on current memory pressure
 * - **Access Pattern Adaptation**: Modifies caching strategy based on detected patterns
 * - **Performance Monitoring**: Enables enhanced performance monitoring and metrics
 * - **Resource Balancing**: Balances memory usage vs. performance trade-offs
 *
 * ## When to Use
 * 
 * This optimization should be called in the following scenarios:
 * - **After Initial Loading**: Once the dataset is fully loaded and initial access patterns are established
 * - **Periodic Optimization**: Periodically during long-running training sessions
 * - **Pattern Changes**: When access patterns change significantly
 * - **Memory Pressure**: When system memory pressure requires cache optimization
 * - **Performance Issues**: When cache performance metrics indicate suboptimal behavior
 *
 * ## Performance Impact
 * 
 * The optimization process itself has minimal performance impact:
 * - **Non-blocking**: Optimization runs in the background without blocking access
 * - **Incremental**: Changes are applied incrementally to avoid disruption
 * - **Adaptive**: Optimization intensity adapts to current system load
 * - **Reversible**: Changes can be reverted if they don't improve performance
 *
 * ## Usage Examples
 *
 * ### Basic Optimization
 * ```c
 * // Optimize after loading
 * struct llama_dataset* dataset = llama_dataset_from_txt(&params, model);
 * if (dataset && llama_dataset_is_streaming_enabled(dataset)) {
 *     if (!llama_dataset_optimize_text_sequence_cache(dataset)) {
 *         fprintf(stderr, "Cache optimization failed: %s\n", llama_dataset_get_error_message());
 *     }
 * }
 * ```
 *
 * ### Periodic Optimization
 * ```c
 * // Optimize periodically during training
 * static int optimization_counter = 0;
 * if (++optimization_counter % 1000 == 0) {  // Every 1000 sequences
 *     llama_dataset_optimize_text_sequence_cache(dataset);
 * }
 * ```
 *
 * ### Performance-driven Optimization
 * ```c
 * // Optimize based on cache performance
 * double hit_ratio;
 * size_t memory_usage, entry_count;
 * if (llama_dataset_get_streaming_stats(dataset, &hit_ratio, &memory_usage, &entry_count)) {
 *     if (hit_ratio < 0.8) {  // Less than 80% hit ratio
 *         llama_dataset_optimize_text_sequence_cache(dataset);
 *     }
 * }
 * ```
 *
 * @param dataset Text dataset to optimize. Must be a valid dataset created with
 *                llama_dataset_from_txt() or llama_dataset_load_text_internal().
 *                The dataset must have streaming enabled for optimization to be
 *                effective. Non-streaming datasets will return success without
 *                performing any operations.
 *
 * @return true on successful optimization or if no optimization was needed,
 *         false on error. Error details are available via llama_dataset_get_error().
 *         Common error conditions include:
 *         - Invalid dataset parameter (NULL or corrupted)
 *         - Dataset not created from text format
 *         - Insufficient memory for optimization operations
 *         - I/O errors during cache reorganization
 *
 * @note This function is thread-safe and can be called concurrently with read
 *       operations on the dataset. However, it should not be called concurrently
 *       with other optimization or configuration operations.
 *
 * @see llama_dataset_set_streaming_cache_size() for cache size configuration
 * @see llama_dataset_get_streaming_stats() for performance monitoring
 * @see llama_dataset_set_adaptive_cache_sizing() for automatic optimization
 *
 * @warning Optimization may temporarily increase memory usage during the
 *          reorganization process. Ensure sufficient memory is available.
 */
bool llama_dataset_optimize_text_sequence_cache(struct llama_dataset * dataset);
