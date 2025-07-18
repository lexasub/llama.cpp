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
./dataset_converter [options] <input_file> <output_file>
```

**Options:**

| Flag                | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `-h`, `--help`      | Show the help message and exit.                              |
| `--model MODEL`     | Path to the `llama.cpp` model for tokenization (required for text input). |
| `--streaming`       | Use streaming mode for large datasets.                       |

### Examples

**Convert a text file to GGUF:**

```bash
./dataset_converter --model ./models/7B/ggml-model-f16.gguf --streaming input.txt output.gguf
```

**Convert a Parquet file to GGUF:**

```bash
./dataset_converter --streaming input.parquet output.gguf
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

// Legacy access functions
uint64_t llama_dataset_get_sequence_count(const struct llama_dataset * dataset);
int32_t llama_dataset_get_sequence_length(const struct llama_dataset * dataset, uint64_t index);
const llama_token * llama_dataset_get_sequence(const struct llama_dataset * dataset, uint64_t index);

// Legacy conversion function
bool llama_dataset_save_gguf(struct llama_dataset * dataset, const char * path);
```

## 7. Dependencies

-   **Apache Arrow**: Required for Parquet file support (`LLAMA_PARQUET=ON`).

## 8. Building the Tool

The Dataset Converter is built as part of the main `llama.cpp` project. The build can be enabled or disabled using the `PROJECT_BUILD_DATASET_CONVERTER` option (default is `ON`).

To build the Dataset Converter with Parquet support, enable the `LLAMA_PARQUET` option in your CMake configuration:

```bash
cmake -B build -DPROJECT_BUILD_DATASET_CONVERTER=ON -DLLAMA_PARQUET=ON
cmake --build build
```

## 9. Testing

Run the tests with:

```bash
ctest -R "_unit|_streaming|_integration"
```

## 10. Important Note on `safetensors`

**This tool does not currently support the `safetensors` format.** The focus is on providing a robust pipeline for formats commonly used in large-scale data processing. Future support for `safetensors` may be considered based on community demand.

## 11. Development Notes

For developers working on this codebase:
- See `docs/REMOVED_FILES.md` for information about the previous implementation.
- The new interface is designed to be simpler and more consistent while maintaining backward compatibility.
- Streaming optimization features significantly improve performance for large datasets.
