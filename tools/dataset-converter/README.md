# Dataset Converter for `llama.cpp`

## 1. Overview

The Dataset Converter is a powerful and efficient tool designed to streamline the preparation of training data for `llama.cpp`. It enables the conversion of datasets from common formats, such as plain text and Parquet, into the GGUF format, which is optimized for use within the `llama.cpp` ecosystem.

This tool is intended for developers and researchers who are fine-tuning models or working with custom datasets, providing a robust pipeline for handling large-scale data with minimal overhead.

## 2. The Problem It Solves

As `llama.cpp` continues to evolve, there is a growing need for accessible tools to support model training and fine-tuning. Preparing datasets for training can be a complex and error-prone process, often requiring multiple scripts and manual intervention.

The Dataset Converter addresses this challenge by:

-   **Simplifying Data Ingestion**: Provides a single, unified interface for processing different data formats.
-   **Optimizing for `llama.cpp`**: Converts datasets into the native GGUF format, ensuring compatibility and performance.
-   **Handling Large Datasets**: Incorporates streaming optimizations to process files that are too large to fit into memory.

## 3. Key Features

-   **Multiple Format Support**: Natively handles text files, Parquet files, and GGUF files.
-   **Efficient Tokenization**: Leverages a `llama.cpp` model to tokenize text-based datasets.
-   **Streaming for Large Datasets**: Employs caching and read-ahead buffering to manage memory usage effectively.
-   **C API**: Offers a simple C API for programmatic integration into custom data pipelines.

## 4. Usage

The tool is run from the command line. The basic syntax is as follows:

```bash
./dataset_converter [options] --in-file <input_file> -o <output_file> --streaming
```
### Examples

**Convert a text file to GGUF:**

```bash
./dataset_converter --model ./models/7B/ggml-model-f16.gguf --streaming --in-file input.txt -o output.gguf
```

**Convert a Parquet file to GGUF:**

```bash
./dataset_converter --streaming --in-file input.parquet -o output.gguf --dataset-column tokens
```

### Advanced Parquet Tokenization Examples

**Mixed Schema Handling - Text and Pre-tokenized Columns:**

```bash
# Process Parquet file with both text and token columns
./dataset_converter --streaming --in-file mixed_data.parquet -o output.gguf \
  --text-column "content" --token-column "preprocessed_tokens" \
  --model ./models/7B/ggml-model-f16.gguf
```

**Large File Streaming with Performance Optimization:**

```bash
# Optimize for large datasets with custom cache settings
./dataset_converter --streaming --in-file large_dataset.parquet -o output.gguf \
  --cache-size 512MB --read-ahead-window 1000 --adaptive-cache \
  --memory-pressure-threshold 0.8
```

**Batch Processing with Memory Management:**

```bash
# Process multiple Parquet files with memory constraints
./dataset_converter --streaming --batch-mode \
  --input-dir ./parquet_files/ --output-dir ./gguf_files/ \
  --max-memory 4GB --parallel-workers 4
```

**Schema Analysis and Validation:**

```bash
# Analyze Parquet schema before conversion
./dataset_converter --analyze-schema --in-file dataset.parquet \
  --validate-columns --report-statistics
```

**Custom Tokenization Configuration:**

```bash
# Fine-tune tokenization for specific use cases
./dataset_converter --streaming --in-file dataset.parquet -o output.gguf \
  --tokenizer-config custom_config.json --max-sequence-length 2048 \
  --padding-strategy truncate --special-tokens-handling preserve
```

## 5. Directory Structure

The codebase is organized into the following modules:

- **core/**: Core dataset functionality and common utilities
- **formats/**: Format-specific implementations
  - **gguf/**: GGUF format support
  - **text/**: Text format support
  - **parquet/**: Parquet format support
- **streaming/**: Streaming optimization for large datasets
- **validation/**: Data validation and integrity checking
- **tools/**: Command-line tools and utilities
- **tests/**: Test suite

## 6. API Overview

For advanced use cases, the converter's core functionality is exposed through a C API.

### Modern API

```c
// Loading datasets
struct llama_dataset * from_gguf(const char * path);
struct llama_dataset * from_txt(const char * path, struct llama_model * model);
struct llama_dataset * from_parquet(const char * path);

// Accessing data
uint64_t n_sequences(const struct llama_dataset * dataset);
int32_t sequence_length(const struct llama_dataset * dataset, uint64_t index);
const int32_t * sequence(const struct llama_dataset * dataset, uint64_t index);

// Converting
void to_gguf(struct llama_dataset * dataset, const char * path);

// Cleanup
void llama_dataset_free(struct llama_dataset * dataset);
```

### Streaming Optimization API

```c
// Configure streaming
bool llama_dataset_set_streaming_cache_size(struct llama_dataset * dataset, size_t cache_size_bytes);
bool llama_dataset_set_streaming_read_ahead(struct llama_dataset * dataset, bool enabled, size_t window_size);
bool llama_dataset_set_adaptive_cache_sizing(struct llama_dataset * dataset, bool enabled);

// Get streaming statistics
bool llama_dataset_get_streaming_stats(
    const struct llama_dataset * dataset,
    double * hit_ratio,
    size_t * memory_usage_bytes,
    size_t * entry_count);
```

### Legacy API (for backward compatibility)

```c
// Legacy loading functions
struct llama_dataset * llama_dataset_load_gguf(const char * path, bool streaming);
struct llama_dataset * llama_dataset_load_text(const char * path, struct llama_model * model, bool streaming);
struct llama_dataset * llama_dataset_load_parquet(const char * path, bool streaming);
```

### Detailed API Reference

#### Core Dataset Functions

**`struct llama_dataset * from_parquet(const char * path)`**
- **Parameters**: 
  - `path`: Path to the Parquet file
- **Returns**: Pointer to dataset structure, or NULL on failure
- **Error Handling**: Check return value; use `llama_dataset_get_last_error()` for details
- **Usage**: Primary function for loading Parquet datasets with automatic schema detection

**`uint64_t n_sequences(const struct llama_dataset * dataset)`**
- **Parameters**: 
  - `dataset`: Valid dataset pointer
- **Returns**: Total number of sequences in the dataset
- **Error Handling**: Returns 0 if dataset is NULL or invalid
- **Usage**: Get total sequence count for iteration planning

**`const int32_t * sequence(const struct llama_dataset * dataset, uint64_t index)`**
- **Parameters**: 
  - `dataset`: Valid dataset pointer
  - `index`: Sequence index (0-based)
- **Returns**: Pointer to token array, or NULL if index is out of bounds
- **Error Handling**: Returns NULL for invalid parameters; check bounds with `n_sequences()`
- **Usage**: Access individual tokenized sequences

#### Parquet-Specific Functions

**`bool llama_dataset_analyze_parquet_schema(const char * path, struct parquet_schema_info * info)`**
- **Parameters**: 
  - `path`: Path to Parquet file
  - `info`: Output structure for schema information
- **Returns**: true on success, false on failure
- **Error Handling**: Check return value; `info` structure contains error details
- **Usage**: Analyze Parquet file structure before loading

**`bool llama_dataset_parquet_set_column_mapping(struct llama_dataset * dataset, const char * text_column, const char * token_column)`**
- **Parameters**: 
  - `dataset`: Valid Parquet dataset pointer
  - `text_column`: Name of text column to tokenize (can be NULL)
  - `token_column`: Name of pre-tokenized column (can be NULL)
- **Returns**: true on success, false on failure
- **Error Handling**: At least one column must be specified
- **Usage**: Configure mixed schema handling for datasets with both text and tokens

**`bool llama_dataset_parquet_validate_schema(const struct llama_dataset * dataset)`**
- **Parameters**: 
  - `dataset`: Valid Parquet dataset pointer
- **Returns**: true if schema is valid, false otherwise
- **Error Handling**: Use `llama_dataset_get_validation_errors()` for detailed error information
- **Usage**: Validate schema consistency before processing

#### Tokenization Functions

**`bool llama_dataset_tokenize_text_column(struct llama_dataset * dataset, const char * column_name, struct llama_model * model)`**
- **Parameters**: 
  - `dataset`: Valid dataset pointer
  - `column_name`: Name of text column to tokenize
  - `model`: Tokenization model
- **Returns**: true on success, false on failure
- **Error Handling**: Check model validity and column existence
- **Usage**: Tokenize specific text columns in Parquet datasets

**`bool llama_dataset_get_tokenization_stats(const struct llama_dataset * dataset, struct tokenization_stats * stats)`**
- **Parameters**: 
  - `dataset`: Valid dataset pointer
  - `stats`: Output structure for tokenization statistics
- **Returns**: true on success, false on failure
- **Error Handling**: Ensure dataset has been tokenized
- **Usage**: Get performance metrics and tokenization statistics

#### Streaming Configuration Functions

**`bool llama_dataset_set_streaming_cache_size(struct llama_dataset * dataset, size_t cache_size_bytes)`**
- **Parameters**: 
  - `dataset`: Valid dataset pointer
  - `cache_size_bytes`: Cache size in bytes (minimum 1MB, maximum system-dependent)
- **Returns**: true on success, false on failure
- **Error Handling**: Validates cache size limits and available memory
- **Usage**: Configure cache size for optimal memory usage

**`bool llama_dataset_set_streaming_read_ahead(struct llama_dataset * dataset, bool enabled, size_t window_size)`**
- **Parameters**: 
  - `dataset`: Valid dataset pointer
  - `enabled`: Enable/disable read-ahead buffering
  - `window_size`: Number of sequences to buffer ahead (0 for auto)
- **Returns**: true on success, false on failure
- **Error Handling**: Window size must be reasonable for available memory
- **Usage**: Optimize sequential access patterns

**`bool llama_dataset_set_memory_pressure_threshold(struct llama_dataset * dataset, double threshold)`**
- **Parameters**: 
  - `dataset`: Valid dataset pointer
  - `threshold`: Memory pressure threshold (0.0 to 1.0)
- **Returns**: true on success, false on failure
- **Error Handling**: Threshold must be between 0.1 and 0.95
- **Usage**: Configure automatic cache management under memory pressure

#### Error Handling Functions

**`const char * llama_dataset_get_last_error(void)`**
- **Returns**: String describing the last error, or NULL if no error
- **Usage**: Get detailed error information after API calls fail

**`bool llama_dataset_clear_errors(void)`**
- **Returns**: true on success
- **Usage**: Clear error state for fresh error tracking

## 7. Configuration Guide

### Tokenization Configuration

#### Cache Settings

**Memory Cache Configuration:**
```c
// Set cache size based on available memory
size_t cache_size = 256 * 1024 * 1024;  // 256MB
llama_dataset_set_streaming_cache_size(dataset, cache_size);

// Enable adaptive cache sizing
llama_dataset_set_adaptive_cache_sizing(dataset, true);
```

**Cache Performance Tuning:**
```c
// Configure cache eviction policy
llama_dataset_set_cache_eviction_policy(dataset, LLAMA_CACHE_LRU);

// Set cache warming for predictable access patterns
llama_dataset_set_cache_warming(dataset, true, 100);  // Warm 100 sequences ahead
```

#### Streaming Optimization

**Read-Ahead Configuration:**
```c
// Enable read-ahead for sequential access
llama_dataset_set_streaming_read_ahead(dataset, true, 500);

// Configure read-ahead for random access patterns
llama_dataset_set_random_access_optimization(dataset, true);
```

**Memory Management:**
```c
// Set memory pressure thresholds
llama_dataset_set_memory_pressure_threshold(dataset, 0.8);  // 80% memory usage

// Configure memory monitoring interval
llama_dataset_set_memory_monitor_interval(dataset, 1000);  // Check every 1000ms
```

#### Tokenization Settings

**Model Configuration:**
```c
// Configure tokenization model parameters
struct tokenization_config config = {
    .max_sequence_length = 2048,
    .padding_strategy = LLAMA_PAD_TRUNCATE,
    .special_tokens = LLAMA_SPECIAL_PRESERVE,
    .batch_size = 32
};
llama_dataset_set_tokenization_config(dataset, &config);
```

**Column Mapping for Mixed Schemas:**
```c
// Configure text and token column handling
llama_dataset_parquet_set_column_mapping(dataset, "text_content", "preprocessed_tokens");

// Set column priority for mixed data
llama_dataset_set_column_priority(dataset, LLAMA_PREFER_TOKENS);
```

### Environment Variables

**Memory Configuration:**
```bash
export LLAMA_DATASET_CACHE_SIZE=512MB
export LLAMA_DATASET_MAX_MEMORY=4GB
export LLAMA_DATASET_MEMORY_PRESSURE_THRESHOLD=0.85
```

**Performance Tuning:**
```bash
export LLAMA_DATASET_READ_AHEAD_SIZE=1000
export LLAMA_DATASET_PARALLEL_WORKERS=4
export LLAMA_DATASET_BATCH_SIZE=64
```

**Debug and Logging:**
```bash
export LLAMA_DATASET_LOG_LEVEL=INFO
export LLAMA_DATASET_PROFILE_PERFORMANCE=1
export LLAMA_DATASET_VALIDATE_SCHEMA=1
```

### Configuration Files

**JSON Configuration Example:**
```json
{
  "tokenization": {
    "max_sequence_length": 2048,
    "padding_strategy": "truncate",
    "special_tokens_handling": "preserve",
    "batch_size": 32
  },
  "streaming": {
    "cache_size_mb": 256,
    "read_ahead_window": 500,
    "adaptive_cache": true,
    "memory_pressure_threshold": 0.8
  },
  "parquet": {
    "text_column": "content",
    "token_column": "tokens",
    "validate_schema": true,
    "mixed_schema_handling": "prefer_tokens"
  }
}
```

## 8. Performance Tuning

### Large Dataset Optimization

#### Memory Management Best Practices

**Cache Sizing Guidelines:**
- **Small datasets (< 1GB)**: Use 10-20% of dataset size for cache
- **Medium datasets (1-10GB)**: Use 256MB-1GB cache with adaptive sizing
- **Large datasets (> 10GB)**: Use 1-4GB cache with aggressive eviction

**Memory Pressure Handling:**
```c
// Monitor memory usage and adjust cache dynamically
struct streaming_stats stats;
llama_dataset_get_streaming_stats(dataset, NULL, NULL, NULL);

if (stats.memory_pressure > 0.85) {
    // Reduce cache size under pressure
    size_t new_size = stats.cache_size * 0.7;
    llama_dataset_set_streaming_cache_size(dataset, new_size);
}
```

#### I/O Optimization

**Sequential Access Patterns:**
```c
// Optimize for sequential reading
llama_dataset_set_access_pattern(dataset, LLAMA_ACCESS_SEQUENTIAL);
llama_dataset_set_streaming_read_ahead(dataset, true, 1000);
```

**Random Access Patterns:**
```c
// Optimize for random access
llama_dataset_set_access_pattern(dataset, LLAMA_ACCESS_RANDOM);
llama_dataset_set_cache_eviction_policy(dataset, LLAMA_CACHE_LFU);
```

#### Parallel Processing

**Multi-threaded Tokenization:**
```c
// Configure parallel tokenization
llama_dataset_set_parallel_tokenization(dataset, 4);  // 4 worker threads

// Set thread-safe access mode
llama_dataset_set_thread_safety(dataset, LLAMA_THREAD_SAFE_READ);
```

**Batch Processing:**
```c
// Process sequences in batches for efficiency
struct batch_config batch = {
    .batch_size = 64,
    .prefetch_batches = 2,
    .parallel_batches = true
};
llama_dataset_set_batch_config(dataset, &batch);
```

### Performance Monitoring

#### Metrics Collection

**Cache Performance:**
```c
struct cache_stats cache_stats;
llama_dataset_get_cache_stats(dataset, &cache_stats);

printf("Cache hit ratio: %.2f%%\n", cache_stats.hit_ratio * 100);
printf("Memory usage: %zu MB\n", cache_stats.memory_usage / (1024*1024));
printf("Evictions: %zu\n", cache_stats.eviction_count);
```

**Tokenization Performance:**
```c
struct tokenization_stats tok_stats;
llama_dataset_get_tokenization_stats(dataset, &tok_stats);

printf("Tokenization rate: %.2f tokens/sec\n", tok_stats.tokens_per_second);
printf("Average sequence length: %.1f\n", tok_stats.avg_sequence_length);
printf("Cache efficiency: %.2f%%\n", tok_stats.cache_efficiency * 100);
```

#### Performance Profiling

**Enable Detailed Profiling:**
```c
// Enable performance profiling
llama_dataset_enable_profiling(dataset, true);

// Get detailed timing information
struct performance_profile profile;
llama_dataset_get_performance_profile(dataset, &profile);

printf("I/O time: %.2fms\n", profile.io_time_ms);
printf("Tokenization time: %.2fms\n", profile.tokenization_time_ms);
printf("Cache operations: %.2fms\n", profile.cache_time_ms);
```

### Optimization Strategies

#### For Different Dataset Sizes

**Small Datasets (< 100MB):**
- Disable streaming, load entirely into memory
- Use minimal cache overhead
- Focus on tokenization speed

**Medium Datasets (100MB - 5GB):**
- Enable streaming with moderate cache (256MB-1GB)
- Use read-ahead buffering
- Balance memory usage and performance

**Large Datasets (> 5GB):**
- Aggressive streaming with large cache (1-4GB)
- Enable adaptive cache management
- Use parallel processing where possible

#### For Different Access Patterns

**Sequential Processing:**
```c
llama_dataset_set_access_pattern(dataset, LLAMA_ACCESS_SEQUENTIAL);
llama_dataset_set_streaming_read_ahead(dataset, true, 1000);
llama_dataset_set_cache_eviction_policy(dataset, LLAMA_CACHE_FIFO);
```

**Random Access:**
```c
llama_dataset_set_access_pattern(dataset, LLAMA_ACCESS_RANDOM);
llama_dataset_set_cache_eviction_policy(dataset, LLAMA_CACHE_LRU);
llama_dataset_set_random_access_optimization(dataset, true);
```

**Mixed Access:**
```c
llama_dataset_set_access_pattern(dataset, LLAMA_ACCESS_MIXED);
llama_dataset_set_adaptive_cache_sizing(dataset, true);
llama_dataset_set_cache_eviction_policy(dataset, LLAMA_CACHE_ADAPTIVE);
```

## 9. Troubleshooting

### Common Issues and Solutions

#### Parquet Loading Issues

**Problem: "Failed to load Parquet file"**
```
Error: Could not open Parquet file: /path/to/dataset.parquet
```
**Solutions:**
1. Verify file exists and is readable
2. Check file is valid Parquet format: `parquet-tools schema dataset.parquet`
3. Ensure Apache Arrow libraries are properly installed
4. Check file permissions and disk space

**Problem: "Schema validation failed"**
```
Error: Parquet schema validation failed: No suitable columns found
```
**Solutions:**
1. Analyze schema: `./dataset_converter --analyze-schema --in-file dataset.parquet`
2. Specify column names explicitly: `--text-column "content" --token-column "tokens"`
3. Check column data types are compatible (string for text, list<int32> for tokens)

#### Memory Issues

**Problem: "Out of memory during tokenization"**
```
Error: Failed to allocate memory for tokenization cache
```
**Solutions:**
1. Reduce cache size: `--cache-size 128MB`
2. Enable adaptive cache: `--adaptive-cache`
3. Process in smaller batches: `--batch-size 16`
4. Use streaming mode: `--streaming`

**Problem: "Memory pressure too high"**
```
Warning: Memory pressure at 95%, reducing cache size
```
**Solutions:**
1. Lower memory pressure threshold: `--memory-pressure-threshold 0.7`
2. Increase available system memory
3. Close other memory-intensive applications
4. Use smaller cache size initially

#### Tokenization Issues

**Problem: "Tokenization model not found"**
```
Error: Could not load tokenization model: /path/to/model.gguf
```
**Solutions:**
1. Verify model file exists and is valid GGUF format
2. Check model is compatible with dataset converter
3. Use absolute path to model file
4. Ensure model has proper tokenizer metadata

**Problem: "Mixed schema handling failed"**
```
Error: Cannot process mixed text and token columns
```
**Solutions:**
1. Specify column priority: `--prefer-tokens` or `--prefer-text`
2. Validate column data types match expectations
3. Check for null values in critical columns
4. Use schema analysis to understand data structure

#### Performance Issues

**Problem: "Slow tokenization performance"**
```
Warning: Tokenization rate below 100 tokens/sec
```
**Solutions:**
1. Increase batch size: `--batch-size 64`
2. Enable parallel processing: `--parallel-workers 4`
3. Use larger cache: `--cache-size 512MB`
4. Check system resources (CPU, memory, I/O)

**Problem: "High cache miss rate"**
```
Warning: Cache hit ratio below 50%
```
**Solutions:**
1. Increase cache size if memory allows
2. Adjust access pattern optimization
3. Enable read-ahead buffering
4. Check for memory pressure issues

### Debugging Tools

#### Schema Analysis

**Analyze Parquet file structure:**
```bash
./dataset_converter --analyze-schema --in-file dataset.parquet --verbose
```

**Validate schema compatibility:**
```bash
./dataset_converter --validate-schema --in-file dataset.parquet --report-issues
```

#### Performance Debugging

**Enable detailed logging:**
```bash
export LLAMA_DATASET_LOG_LEVEL=DEBUG
./dataset_converter --streaming --profile-performance --in-file dataset.parquet -o output.gguf
```

**Memory usage monitoring:**
```bash
./dataset_converter --streaming --monitor-memory --memory-report-interval 5000 \
  --in-file dataset.parquet -o output.gguf
```

#### Error Diagnosis

**Common error patterns and solutions:**

1. **Arrow/Parquet library issues:**
   - Reinstall Arrow with proper version compatibility
   - Check CMake configuration: `LLAMA_PARQUET=ON`

2. **File format issues:**
   - Validate Parquet file with external tools
   - Check for corrupted or incomplete files

3. **Memory allocation failures:**
   - Monitor system memory usage
   - Adjust cache and batch sizes
   - Check for memory leaks in long-running processes

4. **Tokenization inconsistencies:**
   - Verify model compatibility
   - Check for special characters or encoding issues
   - Validate tokenizer configuration

### Getting Help

**Enable verbose output for debugging:**
```bash
./dataset_converter --verbose --debug-tokenization --log-cache-stats \
  --in-file dataset.parquet -o output.gguf
```

**Generate diagnostic report:**
```bash
./dataset_converter --generate-diagnostic-report --in-file dataset.parquet \
  --output-report diagnostic.json
```

For additional support, include the diagnostic report and relevant log output when reporting issues.

## 10. Dependencies

-   **Apache Arrow**: Required for Parquet file support (`LLAMA_PARQUET=ON`).

## 11. Building the Tool

The Dataset Converter is built as part of the main `llama.cpp` project. The build can be enabled or disabled using the `PROJECT_BUILD_DATASET_CONVERTER` option (default is `ON`).

To build the Dataset Converter with Parquet support, enable the `LLAMA_PARQUET` option in your CMake configuration:

```bash
cmake -B build -DPROJECT_BUILD_DATASET_CONVERTER=ON -DLLAMA_PARQUET=ON
cmake --build build
```

## 12. Testing

The dataset converter includes comprehensive testing capabilities using CMake custom targets:

### Running Tests

```bash
# Run all CTest tests
ctest -R "_unit|_streaming|_integration"

# Run dataset-specific CTest targets
ctest -R analyze-core-tests          # Core functionality analysis
ctest -R run-monitored-tests         # Streaming functionality tests
ctest -R generate-datasets           # Create comprehensive test datasets
ctest -R run-all-dataset-tests       # Run all dataset converter tests
```

### CMake Test Targets

The following CMake targets replace the original shell scripts:

- `analyze-core-tests`: Analyzes core dataset functionality (replaces analyze-core-tests.sh)
- `run-monitored-tests`: Runs streaming functionality tests with monitoring (replaces run-monitored-tests.sh)
- `generate-datasets`: Creates comprehensive test datasets
- `dataset-test-analysis`: Combined analysis of all functionality
- `setup-dataset-environment`: Sets up testing environment

## 13. Important Note on `safetensors`

**This tool does not currently support the `safetensors` format.** The focus is on providing a robust pipeline for formats commonly used in large-scale data processing. Future support for `safetensors` may be considered based on community demand.

## 14. Development Notes

For developers working on this codebase:
- See `docs/REMOVED_FILES.md` for information about the previous implementation.
- The new interface is designed to be simpler and more consistent while maintaining backward compatibility.
- Streaming optimization features significantly improve performance for large datasets.

Metadata:
```
training.format.version: int16 (e.g. 1000) - Specification version, in case of future changes.

training.format.source: Source format (gguf, text, parquet)

training.dataset.name: string (optional) - Dataset name (e.g. "OpenWebText-ru").

training.dataset.description: string (optional) - Dataset description (e.g. "OpenWebText-ru").

training.dataset.source: string (optional) - URL or description of the data source.

training.file.creation_date: string (ISO 8601) - File creation date.

training.tokenizer.gguf.model: string - Tokenizer model name (llama, gpt2, etc.).

training.tokenizer.gguf.vocab: array[string] - Tokenizer dictionary.

training.tokenizer.gguf.merges: array[string] - Tokenizer merges (for BPE).

training.tokenizer.gguf.pre: string (optional) - Pre-tokenization architecture.

Note: Instead of storing the entire tokenizer, you could reference the model file, but embedding ensures that the data file is completely self-contained.

training.sequence.count: uint64 - Total number of sequences in the file.
```
Tensors:
```
Naming: training.tensor.{index} (e.g. training.tensor.0, training.tensor.1, ...).

Data type: GGML_TYPE_I32 (standard for tokens in llama.cpp).

Shape: [sequence_length] - One-dimensional array. sequence_length will be different for each tensor.
```
