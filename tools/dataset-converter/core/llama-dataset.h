#pragma once

/**
 * @file llama-dataset.h
 * @brief Simple C interface for working with training datasets in different formats.
 *
 * This header provides a clean, simple interface for loading, accessing, and converting
 * training datasets in different formats (GGUF, text, Parquet).
 */

#include "../../ggml/include/gguf.h"
#include "../../ggml/include/ggml.h"
#include "../../include/llama.h"

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Dataset structure for storing and accessing training data.
 *
 * This structure is opaque to the user and should only be accessed through the provided functions.
 * Internally, it contains a GGUF context, a GGML context, and cached tensor pointers for efficient access.
 */
struct llama_dataset;

/**
 * @brief Standard metadata key definitions for dataset properties.
 */
#define DATASET_SEQUENCE_COUNT    "dataset.n_sequences"  // Number of sequences in the dataset
#define DATASET_MAX_LENGTH        "dataset.max_length"   // Maximum sequence length
#define DATASET_SOURCE_FORMAT     "dataset.source"       // Source format (gguf, text, parquet)
#define DATASET_TOKENIZER         "dataset.tokenizer"    // Tokenizer model used
#define DATASET_CREATION_TIME     "dataset.created"      // Creation timestamp

/**
 * @brief Dataset type enumeration for format identification.
 */
enum dataset_type {
    DATASET_GGUF,     // GGUF format (native)
    DATASET_PARQUET,  // Parquet format (requires Arrow/Parquet support)
    DATASET_TEXT      // Text format (requires tokenization)
};

/**
 * @brief Error codes for dataset operations.
 */
enum dataset_error {
    DATASET_SUCCESS = 0,                // No error
    DATASET_ERROR_FILE_NOT_FOUND,       // File not found
    DATASET_ERROR_INVALID_FORMAT,       // Invalid file format
    DATASET_ERROR_MEMORY_ALLOCATION,    // Memory allocation failed
    DATASET_ERROR_TOKENIZATION_FAILED,  // Text tokenization failed
    DATASET_ERROR_STREAMING_NOT_SUPPORTED, // Streaming not supported for this format
    DATASET_ERROR_INVALID_PARAMETER,    // Invalid parameter
    DATASET_ERROR_CONTEXT_CREATION_FAILED, // Context creation failed
    DATASET_ERROR_IO_ERROR              // I/O error
};

//
// Simple procedural interface - core functions
//

/**
 * @brief Load a dataset from a GGUF file.
 *
 * @param path Path to the GGUF file
 * @return Pointer to the dataset, or NULL on error
 */
struct llama_dataset * from_gguf(const char * path);

/**
 * @brief Load a dataset from a text file and tokenize it.
 *
 * @param path Path to the text file
 * @param model Model to use for tokenization
 * @return Pointer to the dataset, or NULL on error
 */
struct llama_dataset * from_txt(const char * path, struct llama_model * model);

/**
 * @brief Load a dataset from a Parquet file.
 *
 * @param path Path to the Parquet file
 * @return Pointer to the dataset, or NULL on error
 */
struct llama_dataset * from_parquet(const char * path);

/**
 * @brief Save a dataset to a GGUF file.
 *
 * @param dataset Dataset to save
 * @param path Path to the output file
 */
void to_gguf(struct llama_dataset * dataset, const char * path);

/**
 * @brief Get the number of sequences in the dataset.
 *
 * @param dataset Dataset to query
 * @return Number of sequences, or 0 if dataset is NULL
 */
uint64_t n_sequences(const struct llama_dataset * dataset);

/**
 * @brief Get the length of a sequence in the dataset.
 *
 * @param dataset Dataset to query
 * @param index Index of the sequence
 * @return Length of the sequence, or 0 if dataset is NULL or index is out of bounds
 */
int32_t sequence_length(const struct llama_dataset * dataset, uint64_t index);

/**
 * @brief Get a pointer to the tokens in a sequence.
 *
 * @param dataset Dataset to query
 * @param index Index of the sequence
 * @return Pointer to the tokens, or NULL if dataset is NULL or index is out of bounds
 */
const int32_t * sequence(const struct llama_dataset * dataset, uint64_t index);

/**
 * @brief Get a pointer to the tensor for a sequence.
 *
 * This is useful for advanced operations that need direct access to the tensor.
 *
 * @param dataset Dataset to query
 * @param index Index of the sequence
 * @return Pointer to the tensor, or NULL if dataset is NULL or index is out of bounds
 */
struct ggml_tensor * sequence_tensor(const struct llama_dataset * dataset, uint64_t index);

/**
 * @brief Free resources associated with a dataset.
 *
 * @param dataset Dataset to free
 */
void llama_dataset_free(struct llama_dataset * dataset);

//
// Metadata access functions
//

/**
 * @brief Get a string metadata value from the dataset.
 *
 * @param dataset Dataset to query
 * @param key Metadata key
 * @return String value, or NULL if not found
 */
const char * dataset_get_metadata_str(const struct llama_dataset * dataset, const char * key);

/**
 * @brief Get an integer metadata value from the dataset.
 *
 * @param dataset Dataset to query
 * @param key Metadata key
 * @param default_value Default value to return if key is not found
 * @return Integer value, or default_value if not found
 */
int64_t dataset_get_metadata_int(const struct llama_dataset * dataset, const char * key, int64_t default_value);

/**
 * @brief Get a float metadata value from the dataset.
 *
 * @param dataset Dataset to query
 * @param key Metadata key
 * @param default_value Default value to return if key is not found
 * @return Float value, or default_value if not found
 */
float dataset_get_metadata_float(const struct llama_dataset * dataset, const char * key, float default_value);

//
// Error handling functions
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

//
// Legacy compatibility functions (for existing code)
//

/**
 * @brief Load a dataset from a GGUF file (legacy function).
 *
 * @param path Path to the GGUF file
 * @param streaming Whether to use streaming mode
 * @return Pointer to the dataset, or NULL on error
 */
struct llama_dataset * llama_dataset_load_gguf(const char * path, bool streaming);

/**
 * @brief Load a dataset from a Parquet file (legacy function).
 *
 * @param path Path to the Parquet file
 * @param streaming Whether to use streaming mode
 * @return Pointer to the dataset, or NULL on error
 */
struct llama_dataset * llama_dataset_load_parquet(const char * path, bool streaming);

/**
 * @brief Save a dataset to a GGUF file (legacy function).
 *
 * @param dataset Dataset to save
 * @param path Path to the output file
 * @return true on success, false on error
 */
bool llama_dataset_save_gguf(struct llama_dataset * dataset, const char * path);

/**
 * @brief Get the number of sequences in the dataset (legacy function).
 *
 * @param dataset Dataset to query
 * @return Number of sequences, or 0 if dataset is NULL
 */
uint64_t llama_dataset_get_sequence_count(const struct llama_dataset * dataset);

/**
 * @brief Get the length of a sequence in the dataset (legacy function).
 *
 * @param dataset Dataset to query
 * @param index Index of the sequence
 * @return Length of the sequence, or 0 if dataset is NULL or index is out of bounds
 */
int32_t llama_dataset_get_sequence_length(const struct llama_dataset * dataset, uint64_t index);

/**
 * @brief Get a pointer to the tokens in a sequence (legacy function).
 *
 * @param dataset Dataset to query
 * @param index Index of the sequence
 * @return Pointer to the tokens, or NULL if dataset is NULL or index is out of bounds
 */
const llama_token * llama_dataset_get_sequence(const struct llama_dataset * dataset, uint64_t index);

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
// Streaming optimization functions
//

/**
 * @brief Configure streaming cache size for a dataset.
 *
 * @param dataset Dataset to configure
 * @param cache_size_bytes Maximum cache size in bytes
 * @return true on success, false on error
 */
bool llama_dataset_set_streaming_cache_size(struct llama_dataset * dataset, size_t cache_size_bytes);

/**
 * @brief Enable or disable read-ahead buffering for streaming.
 *
 * @param dataset Dataset to configure
 * @param enabled Whether to enable read-ahead
 * @param window_size Number of sequences to prefetch (default: 5)
 * @return true on success, false on error
 */
bool llama_dataset_set_streaming_read_ahead(struct llama_dataset * dataset, bool enabled, size_t window_size);

/**
 * @brief Enable or disable adaptive cache sizing based on memory pressure.
 *
 * @param dataset Dataset to configure
 * @param enabled Whether to enable adaptive sizing
 * @return true on success, false on error
 */
bool llama_dataset_set_adaptive_cache_sizing(struct llama_dataset * dataset, bool enabled);

/**
 * @brief Get streaming cache statistics.
 *
 * @param dataset Dataset to query
 * @param hit_ratio Pointer to store hit ratio (0.0-1.0)
 * @param memory_usage_bytes Pointer to store current memory usage
 * @param entry_count Pointer to store number of entries in cache
 * @return true on success, false on error
 */
bool llama_dataset_get_streaming_stats(
    const struct llama_dataset * dataset,
    double * hit_ratio,
    size_t * memory_usage_bytes,
    size_t * entry_count);

#ifdef __cplusplus
}

/**
 * @brief C++ wrapper class for RAII and STL compatibility.
 *
 * This class provides a C++ interface to the dataset API with RAII semantics.
 */
#include <memory>
#include <string>
#include <vector>

class Dataset {
private:
    struct llama_dataset * dataset_;

public:
    // Constructor and destructor
    explicit Dataset(struct llama_dataset * dataset) : dataset_(dataset) {}
    ~Dataset() {
        if (dataset_) {
            llama_dataset_free(dataset_);
        }
    }

    // Move semantics
    Dataset(Dataset&& other) noexcept : dataset_(other.dataset_) {
        other.dataset_ = nullptr;
    }

    Dataset& operator=(Dataset&& other) noexcept {
        if (this != &other) {
            if (dataset_) {
                llama_dataset_free(dataset_);
            }
            dataset_ = other.dataset_;
            other.dataset_ = nullptr;
        }
        return *this;
    }

    // Delete copy constructor and assignment
    Dataset(const Dataset&) = delete;
    Dataset& operator=(const Dataset&) = delete;

    // Static factory methods
    static std::unique_ptr<Dataset> FromGGUF(const std::string& path) {
        auto* dataset = from_gguf(path.c_str());
        return dataset ? std::make_unique<Dataset>(dataset) : nullptr;
    }

    static std::unique_ptr<Dataset> FromText(const std::string& path, llama_model* model) {
        auto* dataset = from_txt(path.c_str(), model);
        return dataset ? std::make_unique<Dataset>(dataset) : nullptr;
    }

    static std::unique_ptr<Dataset> FromParquet(const std::string& path) {
        auto* dataset = from_parquet(path.c_str());
        return dataset ? std::make_unique<Dataset>(dataset) : nullptr;
    }

    // Access methods
    uint64_t GetSequenceCount() const {
        return dataset_ ? n_sequences(dataset_) : 0;
    }

    int32_t GetSequenceLength(uint64_t index) const {
        return dataset_ ? sequence_length(dataset_, index) : 0;
    }

    std::vector<int32_t> GetSequence(uint64_t index) const {
        if (!dataset_) return {};

        const int32_t* tokens = sequence(dataset_, index);
        int32_t length = sequence_length(dataset_, index);

        if (!tokens || length <= 0) return {};

        return std::vector<int32_t>(tokens, tokens + length);
    }

    // Conversion
    void ToGGUF(const std::string& path) const {
        if (dataset_) {
            to_gguf(dataset_, path.c_str());
        }
    }

    // Metadata access
    std::string GetMetadataStr(const std::string& key) const {
        if (!dataset_) return "";
        const char* value = dataset_get_metadata_str(dataset_, key.c_str());
        return value ? std::string(value) : std::string();
    }

    int64_t GetMetadataInt(const std::string& key, int64_t default_value = 0) const {
        return dataset_ ? dataset_get_metadata_int(dataset_, key.c_str(), default_value) : default_value;
    }

    float GetMetadataFloat(const std::string& key, float default_value = 0.0f) const {
        return dataset_ ? dataset_get_metadata_float(dataset_, key.c_str(), default_value) : default_value;
    }

    // Validity check
    bool IsValid() const {
        return dataset_ != nullptr;
    }

    // Access to underlying C structure (for advanced use)
    struct llama_dataset * GetCDataset() const {
        return dataset_;
    }
};
#endif // __cplusplus
