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

## 10. Important Note on `safetensors`

**This tool does not currently support the `safetensors` format.** The focus is on providing a robust pipeline for formats commonly used in large-scale data processing. Future support for `safetensors` may be considered based on community demand.

## 11. Development Notes

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
