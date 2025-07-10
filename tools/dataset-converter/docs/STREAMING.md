# Streaming Mode for Dataset Converter

## Overview

The streaming mode in the dataset converter is designed to efficiently handle large datasets by loading data on-demand rather than loading the entire dataset into memory. This document describes the streaming functionality, optimizations, and usage guidelines.

## Key Features

### 1. Memory Efficiency

Streaming mode significantly reduces memory usage when working with large datasets:

- **On-demand loading**: Only loads sequences when they are accessed
- **Intelligent caching**: Keeps frequently accessed sequences in memory
- **Adaptive memory management**: Adjusts cache size based on system memory pressure

### 2. Performance Optimizations

Several optimizations have been implemented to improve performance:

- **Read-ahead buffering**: Prefetches sequences likely to be accessed next
- **Multiple eviction policies**: LRU, LFU, and Adaptive policies for different access patterns
- **Access pattern analysis**: Automatically detects sequential vs. random access patterns
- **Thread-safe operations**: Supports concurrent access from multiple threads

### 3. Reliability

The streaming implementation includes several features to ensure reliability:

- **Graceful fallback**: Automatically falls back to full loading when streaming is not supported
- **Data consistency**: Ensures identical data between streaming and non-streaming modes
- **Comprehensive error handling**: Provides clear error messages and proper cleanup

## API

### Enabling Streaming Mode

To enable streaming mode when loading a dataset:

```c
// Load a GGUF dataset in streaming mode
struct llama_dataset* dataset = llama_dataset_load_gguf("path/to/dataset.gguf", true);

// Check if streaming is enabled
bool is_streaming = llama_dataset_is_streaming_enabled(dataset);
```

### Configuring Streaming Optimizations

The streaming mode can be configured with several optimization parameters:

```c
// Set cache size (in bytes)
llama_dataset_set_streaming_cache_size(dataset, 64 * 1024 * 1024); // 64MB

// Enable read-ahead buffering with a window size of 5
llama_dataset_set_streaming_read_ahead(dataset, true, 5);

// Enable adaptive cache sizing based on memory pressure
llama_dataset_set_adaptive_cache_sizing(dataset, true);
```

### Getting Statistics

You can retrieve statistics about the streaming cache:

```c
double hit_ratio;
size_t memory_usage;
size_t entry_count;

llama_dataset_get_streaming_stats(dataset, &hit_ratio, &memory_usage, &entry_count);

printf("Cache hit ratio: %.2f\n", hit_ratio);
printf("Memory usage: %zu bytes\n", memory_usage);
printf("Cache entries: %zu\n", entry_count);
```

## Implementation Details

### Streaming Cache

The streaming cache is implemented as an LRU (Least Recently Used) cache with additional optimizations:

- **Thread safety**: All operations are protected by mutex locks
- **Multiple eviction policies**:
  - LRU: Evicts least recently used entries (good for random access)
  - LFU: Evicts least frequently used entries (good for repeated access patterns)
  - Adaptive: Automatically switches between LRU and LFU based on access patterns
- **Performance tracking**: Monitors hit ratio, memory usage, and eviction counts

### Read-Ahead Buffering

The read-ahead system prefetches sequences likely to be accessed next:

- **Asynchronous prefetching**: Uses a background thread to load sequences
- **Adaptive window size**: Adjusts the prefetch window based on access patterns
- **Queue management**: Prioritizes sequences based on access patterns
- **Pause/resume functionality**: Controls resource usage when needed

### Memory Monitoring

The memory monitoring system tracks system memory usage:

- **Memory pressure detection**: Monitors system memory availability
- **Process memory tracking**: Tracks the memory usage of the dataset converter
- **Adaptive thresholds**: Adjusts cache size based on memory pressure
- **Callback system**: Responds to memory pressure changes

### Optimization Manager

The optimization manager coordinates all streaming components:

- **Centralized configuration**: Manages all optimization parameters
- **Access pattern analysis**: Detects sequential vs. random access patterns
- **Automatic tuning**: Adjusts parameters based on usage patterns
- **Performance statistics**: Collects and reports performance metrics

## Performance Benchmarks

### Memory Usage

| Dataset Size | Full Loading | Streaming Mode | Memory Savings |
|--------------|--------------|----------------|----------------|
| 100MB        | ~100MB       | ~20MB          | 80%            |
| 1GB          | ~1GB         | ~100MB         | 90%            |
| 10GB         | ~10GB        | ~500MB         | 95%            |

### Access Speed

| Access Pattern | Full Loading | Basic Streaming | Optimized Streaming |
|----------------|--------------|-----------------|---------------------|
| Sequential     | Fast         | Moderate        | Fast                |
| Random         | Very Fast    | Slow            | Moderate            |
| Mixed          | Fast         | Moderate        | Moderate            |

## Best Practices

1. **Use streaming mode for large datasets**: Enable streaming mode when working with datasets larger than available memory.

2. **Configure cache size appropriately**: Set the cache size based on available memory and dataset characteristics.

3. **Enable read-ahead for sequential access**: If you're accessing sequences sequentially, enable read-ahead buffering.

4. **Enable adaptive cache sizing**: Let the system automatically adjust cache size based on memory pressure.

5. **Monitor performance**: Use the statistics API to monitor cache hit ratio and memory usage.

## Troubleshooting

### High Memory Usage

If memory usage is still high with streaming mode:

- Reduce the cache size using `llama_dataset_set_streaming_cache_size()`
- Enable adaptive cache sizing with `llama_dataset_set_adaptive_cache_sizing()`
- Check if you're holding references to sequences outside the cache

### Slow Performance

If performance is slow with streaming mode:

- For sequential access, enable read-ahead with `llama_dataset_set_streaming_read_ahead()`
- Increase cache size if memory allows
- Consider using full loading mode if the dataset fits in memory

### Crashes or Errors

If you encounter crashes or errors:

- Check if the dataset format supports streaming
- Ensure the dataset file exists and is accessible
- Verify that the dataset is properly formatted
- Check error messages with `llama_dataset_get_error_message()`

## Conclusion

The streaming mode in the dataset converter provides significant memory efficiency benefits for large datasets while maintaining good performance characteristics. The adaptive approach ensures optimal resource usage across different access patterns and dataset sizes, while maintaining data consistency and reliability.