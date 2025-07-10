#include "llama-dataset-parquet.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <vector>

#include "../../common/log.h"
#include "../../ggml/include/ggml.h"
#include "../../ggml/include/gguf.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-utils.h"
#include "llama-impl.h"

#ifdef LLAMA_DATASET_PARQUET_SUPPORT
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>
#endif

/**
 * @brief Parquet-specific format data structure.
 *
 * This structure holds Parquet-specific state for streaming mode.
 */
struct parquet_format_data {
#ifdef LLAMA_DATASET_PARQUET_SUPPORT
    std::shared_ptr<arrow::Table> table;
    std::shared_ptr<parquet::arrow::FileReader> reader;
    std::string file_path;
#endif
    uint64_t n_sequences;
    int32_t max_length;
};



/**
 * @brief Convert Arrow array to token vector.
 *
 * This function converts an Arrow array (typically Int32Array or ListArray)
 * to a vector of tokens that can be stored in a GGML tensor.
 *
 * @param array Arrow array containing token data
 * @param tokens Output vector for tokens
 * @return true on success, false on error
 */
static bool arrow_array_to_tokens(const std::shared_ptr<arrow::Array>& array, std::vector<int32_t>& tokens) {
    if (!array) {
        set_error("Arrow array is null");
        return false;
    }

    // Handle different array types
    switch (array->type_id()) {
        case arrow::Type::INT32: {
            auto int32_array = std::static_pointer_cast<arrow::Int32Array>(array);
            tokens.reserve(int32_array->length());

            for (int64_t i = 0; i < int32_array->length(); i++) {
                if (int32_array->IsNull(i)) {
                    // Skip null values or use a special token
                    continue;
                }
                tokens.push_back(int32_array->Value(i));
            }
            break;
        }

        case arrow::Type::LIST: {
            auto list_array = std::static_pointer_cast<arrow::ListArray>(array);

            // For list arrays, we expect each list to be a sequence of tokens
            // We'll flatten all sequences into a single token vector for now
            // In a more sophisticated implementation, we might handle this differently

            for (int64_t i = 0; i < list_array->length(); i++) {
                if (list_array->IsNull(i)) {
                    continue;
                }

                auto slice = list_array->value_slice(i);
                if (slice->type_id() == arrow::Type::INT32) {
                    auto int32_slice = std::static_pointer_cast<arrow::Int32Array>(slice);
                    for (int64_t j = 0; j < int32_slice->length(); j++) {
                        if (!int32_slice->IsNull(j)) {
                            tokens.push_back(int32_slice->Value(j));
                        }
                    }
                }
            }
            break;
        }

        default:
            set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Unsupported Arrow array type for tokens");
            return false;
    }

    return true;
}

/**
 * @brief Create GGUF context from Parquet data.
 *
 * This function creates a GGUF context and populates it with data from
 * the Parquet file, converting sequences to tensors.
 *
 * @param table Arrow table containing the Parquet data
 * @param dataset Dataset structure to populate
 * @return true on success, false on error
 */
static bool create_gguf_from_parquet(const std::shared_ptr<arrow::Table>& table, struct llama_dataset* dataset) {
    if (!table || !dataset) {
        set_error("Invalid parameters for GGUF creation from Parquet");
        return false;
    }

    // Create GGUF context
    struct gguf_init_params gguf_params = {};
    dataset->ctx = gguf_init_empty();
    if (!dataset->ctx) {
        set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGUF context");
        return false;
    }

    // Find the tokens column
    std::shared_ptr<arrow::ChunkedArray> tokens_column;
    int tokens_column_index = -1;

    for (int i = 0; i < table->num_columns(); i++) {
        std::string column_name = table->schema()->field(i)->name();
        if (column_name == "tokens" || column_name == "input_ids" || column_name == "token_ids") {
            tokens_column = table->column(i);
            tokens_column_index = i;
            break;
        }
    }

    if (!tokens_column || tokens_column_index == -1) {
        set_error_with_code(DATASET_ERROR_INVALID_FORMAT,
            "Parquet file must contain a 'tokens', 'input_ids', or 'token_ids' column");
        return false;
    }

    // Set metadata
    gguf_set_val_str(dataset->ctx, DATASET_SOURCE_FORMAT, "parquet");

    // Get current time for creation timestamp
    time_t now = time(nullptr);
    char time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));
    gguf_set_val_str(dataset->ctx, DATASET_CREATION_TIME, time_str);

    // Process each chunk in the tokens column
    std::vector<std::vector<int32_t>> all_sequences;
    int32_t max_length = 0;

    for (int chunk_idx = 0; chunk_idx < tokens_column->num_chunks(); chunk_idx++) {
        auto chunk = tokens_column->chunk(chunk_idx);

        // Handle different chunk types
        if (chunk->type_id() == arrow::Type::LIST) {
            // Each row is a list of tokens (sequence)
            auto list_array = std::static_pointer_cast<arrow::ListArray>(chunk);

            for (int64_t row = 0; row < list_array->length(); row++) {
                if (list_array->IsNull(row)) {
                    continue;
                }

                std::vector<int32_t> sequence_tokens;
                auto slice = list_array->value_slice(row);

                if (!arrow_array_to_tokens(slice, sequence_tokens)) {
                    LLAMA_LOG_WARN("Failed to convert tokens for sequence %ld in chunk %d", row, chunk_idx);
                    continue;
                }

                if (!sequence_tokens.empty()) {
                    all_sequences.push_back(std::move(sequence_tokens));
                    max_length = std::max(max_length, (int32_t)all_sequences.back().size());
                }
            }
        } else {
            // Assume the entire chunk is one sequence (less common case)
            std::vector<int32_t> sequence_tokens;
            if (arrow_array_to_tokens(chunk, sequence_tokens) && !sequence_tokens.empty()) {
                all_sequences.push_back(std::move(sequence_tokens));
                max_length = std::max(max_length, (int32_t)all_sequences.back().size());
            }
        }
    }

    if (all_sequences.empty()) {
        set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "No valid sequences found in Parquet file");
        return false;
    }

    // Set sequence metadata
    gguf_set_val_u64(dataset->ctx, DATASET_SEQUENCE_COUNT, all_sequences.size());
    gguf_set_val_i32(dataset->ctx, DATASET_MAX_LENGTH, max_length);
    dataset->n_seq = all_sequences.size();

    // Create GGML context for tensor storage (if not streaming)
    if (!dataset->streaming) {
        // Calculate memory requirements
        size_t total_memory = 0;
        for (const auto& seq : all_sequences) {
            total_memory += seq.size() * sizeof(int32_t);
        }
        total_memory += 1024 * 1024; // Add 1MB buffer for metadata and alignment

        struct ggml_init_params ggml_params = {};
        ggml_params.mem_size = total_memory;
        ggml_params.mem_buffer = nullptr;
        ggml_params.no_alloc = false;

        dataset->ggml_ctx = ggml_init(ggml_params);
        if (!dataset->ggml_ctx) {
            set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGML context");
            return false;
        }

        // Create tensors for each sequence
        for (size_t i = 0; i < all_sequences.size(); i++) {
            const auto& sequence = all_sequences[i];

            // Create tensor name
            char tensor_name[64];
            snprintf(tensor_name, sizeof(tensor_name), "sequence_%zu", i);

            // Create tensor
            struct ggml_tensor* tensor = ggml_new_tensor_1d(dataset->ggml_ctx, GGML_TYPE_I32, sequence.size());
            if (!tensor) {
                set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to create tensor for sequence");
                return false;
            }

            ggml_set_name(tensor, tensor_name);

            // Copy data to tensor
            if (tensor->data) {
                memcpy(tensor->data, sequence.data(), sequence.size() * sizeof(int32_t));
            }

            // Add tensor to GGUF context
            gguf_add_tensor(dataset->ctx, tensor);
        }
    } else {
        // In streaming mode, create a minimal GGML context for tensor metadata
        struct ggml_init_params ggml_params = {};
        ggml_params.mem_size = 1024 * 1024; // 1MB for metadata only
        ggml_params.mem_buffer = nullptr;
        ggml_params.no_alloc = true; // Don't allocate tensor data

        dataset->ggml_ctx = ggml_init(ggml_params);
        if (!dataset->ggml_ctx) {
            set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGML context for streaming");
            return false;
        }

        // Create tensor metadata without data for streaming mode
        for (size_t i = 0; i < all_sequences.size(); i++) {
            const auto& sequence = all_sequences[i];

            char tensor_name[64];
            snprintf(tensor_name, sizeof(tensor_name), "sequence_%zu", i);

            // Create tensor without data allocation
            struct ggml_tensor* tensor = ggml_new_tensor_1d(dataset->ggml_ctx, GGML_TYPE_I32, sequence.size());
            if (!tensor) {
                set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to create tensor metadata for streaming");
                return false;
            }

            ggml_set_name(tensor, tensor_name);

            // Add tensor to GGUF context (metadata only)
            gguf_add_tensor(dataset->ctx, tensor);
        }

        // Store sequences in format_data for streaming access
        auto* format_data = (struct parquet_format_data*)dataset->format_data;
        if (format_data) {
            format_data->n_sequences = all_sequences.size();
            format_data->max_length = max_length;

            // Store the actual sequence data for streaming access
            // We'll need this when sequences are requested
            // For now, we'll store it in a simple way
        }
    }

    LLAMA_LOG_INFO("Successfully loaded %zu sequences from Parquet file (max_length=%d)",
                   all_sequences.size(), max_length);

    return true;
}

/**
 * @brief Get tensor data from a Parquet file in streaming mode.
 *
 * This function loads tensor data from a Parquet file on demand in streaming mode.
 * It is used internally by the sequence() function.
 *
 * @param dataset Dataset to query
 * @param index Index of the tensor
 * @return Pointer to the tensor data, or NULL on error
 */
void* get_parquet_tensor_data_streaming(const struct llama_dataset* dataset, uint64_t index) {
    if (!dataset || !dataset->format_data || !dataset->streaming) {
        set_error("Invalid parameters for streaming data access");
        return nullptr;
    }

    auto* format_data = (struct parquet_format_data*)dataset->format_data;
    if (!format_data->table || !format_data->reader) {
        set_error("Parquet data not available for streaming");
        return nullptr;
    }

    // Find the tokens column
    std::shared_ptr<arrow::ChunkedArray> tokens_column;
    int tokens_column_index = -1;

    for (int i = 0; i < format_data->table->num_columns(); i++) {
        std::string column_name = format_data->table->schema()->field(i)->name();
        if (column_name == "tokens" || column_name == "input_ids" || column_name == "token_ids") {
            tokens_column = format_data->table->column(i);
            tokens_column_index = i;
            break;
        }
    }

    if (!tokens_column || tokens_column_index == -1) {
        set_error("Tokens column not found in Parquet table");
        return nullptr;
    }

    // Get sequence length from tensor metadata
    int32_t seq_length = sequence_length(dataset, index);
    if (seq_length <= 0) {
        set_error("Invalid sequence length for streaming data access");
        return nullptr;
    }

    // Allocate memory for the sequence data
    int32_t* data = (int32_t*)malloc(seq_length * sizeof(int32_t));
    if (!data) {
        set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION, "Failed to allocate memory for sequence data");
        return nullptr;
    }

    // Extract the sequence data from the Parquet file
    try {
        // This is a simplified implementation that assumes each row is a sequence
        // In a real implementation, we would need to handle different Parquet schemas

        // For list arrays, we need to find the correct row
        if (tokens_column->chunk(0)->type_id() == arrow::Type::LIST) {
            // Find the chunk and row that contains our sequence
            int64_t row_count = 0;
            int chunk_idx = 0;
            int64_t row_in_chunk = 0;

            // Skip to the correct chunk and row
            for (chunk_idx = 0; chunk_idx < tokens_column->num_chunks(); chunk_idx++) {
                auto chunk = tokens_column->chunk(chunk_idx);
                if (row_count + chunk->length() > index) {
                    row_in_chunk = index - row_count;
                    break;
                }
                row_count += chunk->length();
            }

            if (chunk_idx >= tokens_column->num_chunks()) {
                set_error("Sequence index out of bounds for streaming");
                free(data);
                return nullptr;
            }

            auto chunk = tokens_column->chunk(chunk_idx);
            auto list_array = std::static_pointer_cast<arrow::ListArray>(chunk);

            if (row_in_chunk >= list_array->length() || list_array->IsNull(row_in_chunk)) {
                set_error("Invalid row for sequence data");
                free(data);
                return nullptr;
            }

            auto slice = list_array->value_slice(row_in_chunk);
            if (slice->type_id() != arrow::Type::INT32) {
                set_error("Unexpected value type in list array");
                free(data);
                return nullptr;
            }

            auto int32_slice = std::static_pointer_cast<arrow::Int32Array>(slice);
            if (int32_slice->length() != seq_length) {
                set_error("Sequence length mismatch in streaming mode");
                free(data);
                return nullptr;
            }

            // Copy the data
            for (int64_t i = 0; i < int32_slice->length(); i++) {
                data[i] = int32_slice->IsNull(i) ? 0 : int32_slice->Value(i);
            }
        } else {
            // For flat arrays, we need to extract a range of values
            // This is less common for sequence data
            set_error("Flat array streaming not implemented yet");
            free(data);
            return nullptr;
        }

        return data;
    } catch (const std::exception& e) {
        set_error(("Streaming data access failed: " + std::string(e.what())).c_str());
        free(data);
        return nullptr;
    }
}

/**
 * @brief Load a dataset from a Parquet file (internal implementation).
 */
struct llama_dataset * llama_dataset_load_parquet_internal(const char * path, bool streaming) {
    if (!path) {
        set_error_with_code(DATASET_ERROR_INVALID_PARAMETER, "Path cannot be null");
        return nullptr;
    }

#ifndef LLAMA_DATASET_PARQUET_SUPPORT
    set_error_with_code(DATASET_ERROR_INVALID_FORMAT,
        "Parquet support not enabled. Rebuild with LLAMA_PARQUET=ON");
    return nullptr;
#else
    // Check if file exists
    FILE* file = fopen(path, "rb");
    if (!file) {
        set_error_with_code(DATASET_ERROR_FILE_NOT_FOUND, "Parquet file not found");
        return nullptr;
    }
    fclose(file);

    // Create dataset structure
    struct llama_dataset * dataset = dataset_alloc(DATASET_PARQUET, streaming);
    if (!dataset) {
        return nullptr; // Error already set by dataset_alloc
    }

    // Allocate format-specific data
    auto* format_data = new parquet_format_data();
    format_data->file_path = std::string(path);
    format_data->n_sequences = 0;
    format_data->max_length = 0;
    dataset->format_data = format_data;

    try {
        // Open Parquet file
        std::shared_ptr<arrow::io::ReadableFile> infile;
        auto result = arrow::io::ReadableFile::Open(path);
        if (!result.ok()) {
            set_error_with_code(DATASET_ERROR_IO_ERROR,
                ("Failed to open Parquet file: " + result.status().ToString()).c_str());
            llama_dataset_free(dataset);
            return nullptr;
        }
        infile = result.ValueOrDie();

        // Create Parquet reader
        std::unique_ptr<parquet::arrow::FileReader> reader;
        auto status = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader);
        if (!status.ok()) {
            set_error_with_code(DATASET_ERROR_INVALID_FORMAT,
                ("Failed to create Parquet reader: " + status.ToString()).c_str());
            llama_dataset_free(dataset);
            return nullptr;
        }

        format_data->reader = std::shared_ptr<parquet::arrow::FileReader>(reader.release());

        // Read table
        std::shared_ptr<arrow::Table> table;
        status = format_data->reader->ReadTable(&table);
        if (!status.ok()) {
            set_error_with_code(DATASET_ERROR_IO_ERROR,
                ("Failed to read Parquet table: " + status.ToString()).c_str());
            llama_dataset_free(dataset);
            return nullptr;
        }

        format_data->table = table;

        // Validate schema
        if (!validate_parquet_schema(path)) {
            // Error already set by validate_parquet_schema
            llama_dataset_free(dataset);
            return nullptr;
        }

        // Create GGUF context from Parquet data
        if (!create_gguf_from_parquet(table, dataset)) {
            // Error already set by create_gguf_from_parquet
            llama_dataset_free(dataset);
            return nullptr;
        }

        // Cache tensors for efficient access
        if (!dataset_cache_tensors(dataset)) {
            // Error already set by dataset_cache_tensors
            llama_dataset_free(dataset);
            return nullptr;
        }

        LLAMA_LOG_INFO("Successfully loaded Parquet dataset from %s (%zu sequences, streaming=%s)",
                       path, dataset->n_seq, streaming ? "true" : "false");

        return dataset;

    } catch (const std::exception& e) {
        set_error_with_code(DATASET_ERROR_INVALID_FORMAT,
            ("Parquet loading failed: " + std::string(e.what())).c_str());
        llama_dataset_free(dataset);
        return nullptr;
    }
#endif // LLAMA_DATASET_PARQUET_SUPPORT
}

/**
 * @brief Validate Parquet file schema for dataset compatibility.
 */
bool validate_parquet_schema(const char * path) {
    if (!path) {
        set_error("Path cannot be null for schema validation");
        return false;
    }

#ifndef LLAMA_DATASET_PARQUET_SUPPORT
    set_error("Parquet support not enabled");
    return false;
#else
    try {
        // Open file
        std::shared_ptr<arrow::io::ReadableFile> infile;
        auto result = arrow::io::ReadableFile::Open(path);
        if (!result.ok()) {
            set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to open Parquet file for validation");
            return false;
        }
        infile = result.ValueOrDie();

        // Create reader
        std::unique_ptr<parquet::arrow::FileReader> reader;
        auto status = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader);
        if (!status.ok()) {
            set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to create Parquet reader for validation");
            return false;
        }

        // Get schema
        std::shared_ptr<arrow::Schema> schema;
        status = reader->GetSchema(&schema);
        if (!status.ok()) {
            set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to read Parquet schema");
            return false;
        }

        // Check for required columns
        bool has_tokens_column = false;
        for (int i = 0; i < schema->num_fields(); i++) {
            std::string field_name = schema->field(i)->name();
            auto field_type = schema->field(i)->type();

            if (field_name == "tokens" || field_name == "input_ids" || field_name == "token_ids") {
                has_tokens_column = true;

                // Validate column type
                if (field_type->id() == arrow::Type::LIST) {
                    // List of integers (most common case)
                    auto list_type = std::static_pointer_cast<arrow::ListType>(field_type);
                    if (list_type->value_type()->id() != arrow::Type::INT32) {
                        set_error_with_code(DATASET_ERROR_INVALID_FORMAT,
                            "Tokens column must contain lists of int32 values");
                        return false;
                    }
                } else if (field_type->id() != arrow::Type::INT32) {
                    set_error_with_code(DATASET_ERROR_INVALID_FORMAT,
                        "Tokens column must be int32 or list<int32>");
                    return false;
                }
                break;
            }
        }

        if (!has_tokens_column) {
            set_error_with_code(DATASET_ERROR_INVALID_FORMAT,
                "Parquet file must contain a 'tokens', 'input_ids', or 'token_ids' column");
            return false;
        }

        return true;

    } catch (const std::exception& e) {
        set_error_with_code(DATASET_ERROR_INVALID_FORMAT,
            ("Schema validation failed: " + std::string(e.what())).c_str());
        return false;
    }
#endif // LLAMA_DATASET_PARQUET_SUPPORT
}

/**
 * @brief Get Parquet file metadata for dataset information.
 */
bool get_parquet_metadata(const char * path, uint64_t * n_sequences, int32_t * max_length) {
    if (!path || !n_sequences || !max_length) {
        set_error("Invalid parameters for metadata extraction");
        return false;
    }

    *n_sequences = 0;
    *max_length = 0;

#ifndef LLAMA_DATASET_PARQUET_SUPPORT
    set_error("Parquet support not enabled");
    return false;
#else
    try {
        // Open file
        std::shared_ptr<arrow::io::ReadableFile> infile;
        auto result = arrow::io::ReadableFile::Open(path);
        if (!result.ok()) {
            set_error_with_code(DATASET_ERROR_IO_ERROR, "Failed to open Parquet file for metadata");
            return false;
        }
        infile = result.ValueOrDie();

        // Create reader
        std::unique_ptr<parquet::arrow::FileReader> reader;
        auto status = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader);
        if (!status.ok()) {
            set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Failed to create Parquet reader for metadata");
            return false;
        }

        // Get basic metadata
        auto parquet_metadata = reader->parquet_reader()->metadata();
        *n_sequences = parquet_metadata->num_rows();

        // For max_length, we'd need to read the data, which is expensive
        // For now, we'll set a reasonable default and let the actual loading determine it
        *max_length = 2048; // Default assumption

        return true;

    } catch (const std::exception& e) {
        set_error_with_code(DATASET_ERROR_INVALID_FORMAT,
            ("Metadata extraction failed: " + std::string(e.what())).c_str());
        return false;
    }
#endif // LLAMA_DATASET_PARQUET_SUPPORT
}
