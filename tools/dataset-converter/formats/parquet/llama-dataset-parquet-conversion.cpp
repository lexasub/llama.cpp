/**
 * @file llama-dataset-parquet-conversion.cpp
 * @brief Parquet data conversion functionality for llama.cpp dataset handling.
 *
 * This file contains functions for converting Parquet data to GGUF format,
 * including Arrow array to token conversion, tensor creation, and streaming
 * data access. Split from main parquet implementation for better modularity.
 */

#include "llama-model.h"
#ifdef LLAMA_PARQUET
#include "llama-dataset-parquet.h"
#include "llama-dataset-parquet-internal.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <cstdio>
#include <cstring>
#include <ctime>
#include <memory>
#include <vector>
#include <string>
#include <algorithm>

#include "common/common.h"
#include "ggml/include/ggml.h"
#include "ggml/include/gguf.h"
#include "llama-dataset-internal.h"
#include "llama-dataset-utils.h"
#include "llama.h"
#include "llama-impl.h"
static void handle_tokenization_error(
    int64_t row_index,
    const std::string & column_name,
    const std::string & text_content,
    const char * error_message = nullptr
);
static bool process_mixed_content_parquet(
    const std::shared_ptr<arrow::Table> & table,
    struct llama_dataset * dataset,
    struct parquet_format_data * format_data,
    std::vector<std::vector<int32_t>> & all_sequences,
    int32_t & max_length
);
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
bool llama_dataset_arrow_array_to_tokens(const std::shared_ptr<arrow::Array> & array,
                                                std::vector<int32_t> &                tokens) {
    if (!array) {
        llama_dataset_set_error("Arrow array is null");
        return false;
    }

    // Handle different array types
    switch (array->type_id()) {
        case arrow::Type::INT32:
            {
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

        case arrow::Type::LIST:
            {
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
            llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "Unsupported Arrow array type for tokens");
            return false;
    }

    return true;
}

/**
 * @brief Create GGUF context from Parquet data with tokenization support.
 *
 * This function creates a GGUF context and populates it with data from
 * the Parquet file, converting sequences to tensors. It supports both
 * pre-tokenized data and raw text that needs to be tokenized.
 *
 * @param table Arrow table containing the Parquet data
 * @param dataset Dataset structure to populate
 * @return true on success, false on error
 */
bool llama_dataset_create_gguf_from_parquet(const std::shared_ptr<arrow::Table> & table,
                                                   struct llama_dataset * dataset) {
    if (!table || !dataset) {
        llama_dataset_set_error("Invalid parameters for GGUF creation from Parquet");
        return false;
    }

    // Create GGUF context
    dataset->ctx = gguf_init_empty();
    if (!dataset->ctx) {
        llama_dataset_set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGUF context");
        return false;
    }

    // Get format data for tokenization support
    auto * format_data = static_cast<struct parquet_format_data *>(dataset->format_data);
    if (!format_data) {
        llama_dataset_set_error("Format data not available for Parquet processing");
        return false;
    }

    // Find the primary data column (tokens or text)
    std::shared_ptr<arrow::ChunkedArray> data_column;
    int data_column_index = -1;
    bool is_text_column = false;

    // First try to find the specified column
    for (int i = 0; i < table->num_columns(); i++) {
        std::string column_name = table->schema()->field(i)->name();
        if (column_name == dataset->column) {
            data_column = table->column(i);
            data_column_index = i;

            // Check if this is a text column that needs tokenization
            auto field_type = table->schema()->field(i)->type();
            is_text_column = is_text_column_type(field_type);
            break;
        }
    }

    // If not found, try to use schema analysis results
    if (!data_column && format_data->schema_analyzed) {
        if (format_data->tokenizer && format_data->schema_info.primary_text_column_index >= 0) {
            // Use primary text column for tokenization
            data_column_index = format_data->schema_info.primary_text_column_index;
            data_column = table->column(data_column_index);
            is_text_column = true;
        } else if (format_data->schema_info.primary_token_column_index >= 0) {
            // Use primary token column
            data_column_index = format_data->schema_info.primary_token_column_index;
            data_column = table->column(data_column_index);
            is_text_column = false;
        }
    }

    if (!data_column || data_column_index == -1) {
        const auto msg{"Parquet file must contain " + dataset->column + " column or compatible data column"};
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, msg.c_str());
        return false;
    }

    // Set metadata
    gguf_set_val_str(dataset->ctx, TRAINING_FORMAT_SOURCE, "parquet");

    // Get current time for creation timestamp
    time_t now = time(nullptr);
    char   time_str[64];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", localtime(&now));
    gguf_set_val_str(dataset->ctx, TRAINING_CREATION_TIME, time_str);

    // Add tokenization metadata if applicable
    if (is_text_column && format_data->tokenizer) {
        gguf_set_val_str(dataset->ctx, "training.tokenization.source", "text");
        gguf_set_val_str(dataset->ctx, "training.tokenization.column",
                        table->schema()->field(data_column_index)->name().c_str());
    } else if (format_data->mixed_content) {
        gguf_set_val_str(dataset->ctx, "training.tokenization.source", "mixed");
    } else {
        gguf_set_val_str(dataset->ctx, "training.tokenization.source", "pre-tokenized");
    }

    // Process each chunk in the data column (tokens or text)
    std::vector<std::vector<int32_t>> all_sequences;
    int32_t max_length = 0;

    for (int chunk_idx = 0; chunk_idx < data_column->num_chunks(); chunk_idx++) {
        auto chunk = data_column->chunk(chunk_idx);

        if (is_text_column) {
            // Handle text data that needs tokenization
            if (!format_data->tokenizer) {
                llama_dataset_set_error("Text column found but tokenizer not available");
                return false;
            }

            if (chunk->type_id() == arrow::Type::STRING || chunk->type_id() == arrow::Type::LARGE_STRING) {
                auto string_array = std::static_pointer_cast<arrow::StringArray>(chunk);

                for (int64_t row = 0; row < string_array->length(); row++) {
                    if (string_array->IsNull(row)) {
                        continue;
                    }

                    std::string text = string_array->GetString(row);
                    if (text.empty()) {
                        continue;
                    }

                    // Tokenize the text
                    std::vector<int32_t> sequence_tokens = format_data->tokenizer->tokenize_text(text);
                    if (sequence_tokens.empty()) {
                        handle_tokenization_error(row, "text", text, "Empty token sequence returned");
                        continue;
                    }

                    all_sequences.push_back(std::move(sequence_tokens));
                    max_length = std::max(max_length, static_cast<int32_t>(all_sequences.back().size()));
                }
            } else {
                LLAMA_LOG_WARN("Unsupported text column type in chunk %d\n", chunk_idx);
                continue;
            }
        } else {
            // Handle pre-tokenized data
            if (chunk->type_id() == arrow::Type::LIST) {
                // Each row is a list of tokens (sequence)
                auto list_array = std::static_pointer_cast<arrow::ListArray>(chunk);

                for (int64_t row = 0; row < list_array->length(); row++) {
                    if (list_array->IsNull(row)) {
                        continue;
                    }

                    std::vector<int32_t> sequence_tokens;
                    auto slice = list_array->value_slice(row);

                    if (!llama_dataset_arrow_array_to_tokens(slice, sequence_tokens)) {
                        LLAMA_LOG_WARN("Failed to convert tokens for sequence %ld in chunk %d\n", row, chunk_idx);
                        continue;
                    }

                    if (!sequence_tokens.empty()) {
                        all_sequences.push_back(std::move(sequence_tokens));
                        max_length = std::max(max_length, static_cast<int32_t>(all_sequences.back().size()));
                    }
                }
            } else {
                // Assume the entire chunk is one sequence (less common case)
                std::vector<int32_t> sequence_tokens;
                if (llama_dataset_arrow_array_to_tokens(chunk, sequence_tokens) && !sequence_tokens.empty()) {
                    all_sequences.push_back(std::move(sequence_tokens));
                    max_length = std::max(max_length, static_cast<int32_t>(all_sequences.back().size()));
                }
            }
        }
    }

    // Handle mixed content if detected
    if (format_data->mixed_content && all_sequences.empty()) {
        LLAMA_LOG_INFO("Processing mixed content Parquet file\n");
        if (!process_mixed_content_parquet(table, dataset, format_data, all_sequences, max_length)) {
            // Error already set by process_mixed_content_parquet
            return false;
        }
    }

    if (all_sequences.empty()) {
        llama_dataset_set_error_with_code(DATASET_ERROR_INVALID_FORMAT, "No valid sequences found in Parquet file");
        return false;
    }

    // Set sequence metadata
    gguf_set_val_u64(dataset->ctx, TRAINING_SEQUENCE_COUNT, all_sequences.size());
    gguf_set_val_i32(dataset->ctx, TRAINING_MAX_LENGTH, max_length);
    dataset->n_seq = all_sequences.size();

    // Add tokenization statistics if tokenizer was used
    if (format_data->tokenizer) {
        size_t total_tokens = format_data->tokenizer->get_total_tokens();
        size_t unique_texts = format_data->tokenizer->get_unique_texts();
        double cache_hit_ratio = format_data->tokenizer->get_cache_hit_ratio();

        gguf_set_val_u64(dataset->ctx, "training.tokenization.total_tokens", total_tokens);
        gguf_set_val_u64(dataset->ctx, "training.tokenization.unique_texts", unique_texts);
        gguf_set_val_f32(dataset->ctx, "training.tokenization.cache_hit_ratio", static_cast<float>(cache_hit_ratio));

        LLAMA_LOG_INFO("Tokenization stats: %zu total tokens, %zu unique texts, %.2f%% cache hit ratio\n",
                      total_tokens, unique_texts, cache_hit_ratio * 100.0);
    }

    // Create GGML context for tensor storage (if not streaming)
    if (!dataset->streaming) {
        // Calculate memory requirements
        size_t total_memory = 0;
        for (const auto & seq : all_sequences) {
            total_memory += seq.size() * sizeof(int32_t);
        }
        total_memory += 1024 * 1024;  // Add 1MB buffer for metadata and alignment

        struct ggml_init_params ggml_params = {};
        ggml_params.mem_size                = total_memory;
        ggml_params.mem_buffer              = nullptr;
        ggml_params.no_alloc                = false;

        dataset->ggml_ctx = ggml_init(ggml_params);
        if (!dataset->ggml_ctx) {
            llama_dataset_set_error_with_code(DATASET_ERROR_CONTEXT_CREATION_FAILED, "Failed to create GGML context");
            return false;
        }

        // Create tensors for each sequence
        for (size_t i = 0; i < all_sequences.size(); i++) {
            const auto & sequence = all_sequences[i];

            // Create tensor name
            char tensor_name[64];
            snprintf(tensor_name, sizeof(tensor_name), "sequence_%zu", i);

            // Create tensor
            struct ggml_tensor * tensor = ggml_new_tensor_1d(dataset->ggml_ctx, GGML_TYPE_I32, sequence.size());
            if (!tensor) {
                llama_dataset_set_error_with_code(DATASET_ERROR_MEMORY_ALLOCATION,
                                                  "Failed to create tensor for sequence");
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
        // Setup streaming mode using dedicated streaming module
        if (!llama_dataset_setup_streaming_mode(dataset, all_sequences, max_length)) {
            // Error already set by setup_streaming_mode
            return false;
        }
    }

    LLAMA_LOG_INFO("Successfully loaded %zu sequences from Parquet file (max_length=%d)\n", all_sequences.size(),
                   max_length);

    return true;
}

/**
 * @brief Handle tokenization failure with detailed error reporting.
 *
 * This function provides detailed error reporting for tokenization failures,
 * including the specific row, column, and text content that failed to tokenize.
 *
 * @param row_index Row number where tokenization failed
 * @param column_name Name of the column being processed
 * @param text_content Text content that failed to tokenize (truncated for logging)
 * @param error_message Additional error message from tokenizer
 */
static void handle_tokenization_error(
    int64_t row_index,
    const std::string & column_name,
    const std::string & text_content,
    const char * error_message
) {
    std::string truncated_text = text_content.length() > 100 ?
        text_content.substr(0, 100) + "..." : text_content;

    if (error_message) {
        LLAMA_LOG_ERROR("Tokenization failed at row %ld, column '%s': %s. Text: '%s'\n",
                       row_index, column_name.c_str(), error_message, truncated_text.c_str());
    } else {
        LLAMA_LOG_ERROR("Tokenization failed at row %ld, column '%s'. Text: '%s'\n",
                       row_index, column_name.c_str(), truncated_text.c_str());
    }
}

/**
 * @brief Process mixed content Parquet data (both text and pre-tokenized columns).
 *
 * This function handles Parquet files that contain both text columns requiring
 * tokenization and pre-tokenized columns. It prioritizes tokenized data when
 * available and falls back to text tokenization when needed.
 *
 * @param table Arrow table containing mixed content
 * @param dataset Dataset structure to populate
 * @param format_data Parquet format data with schema information
 * @param all_sequences Output vector for all processed sequences
 * @param max_length Output for maximum sequence length
 * @return true on success, false on error
 */
static bool process_mixed_content_parquet(
    const std::shared_ptr<arrow::Table> & table,
    struct llama_dataset * dataset,
    struct parquet_format_data * format_data,
    std::vector<std::vector<int32_t>> & all_sequences,
    int32_t & max_length
) {
    if (!table || !dataset || !format_data) {
        llama_dataset_set_error("Invalid parameters for mixed content processing");
        return false;
    }

    // Process each row, checking for both text and token columns
    int64_t num_rows = table->num_rows();

    for (int64_t row = 0; row < num_rows; row++) {
        std::vector<int32_t> sequence_tokens;
        bool sequence_processed = false;

        // First, try to use pre-tokenized data if available
        if (format_data->schema_info.primary_token_column_index >= 0) {
            auto token_column = table->column(format_data->schema_info.primary_token_column_index);

            // Get the token data for this row
            for (int chunk_idx = 0; chunk_idx < token_column->num_chunks(); chunk_idx++) {
                auto chunk = token_column->chunk(chunk_idx);

                if (chunk->type_id() == arrow::Type::LIST) {
                    auto list_array = std::static_pointer_cast<arrow::ListArray>(chunk);

                    // Find the row within this chunk
                    int64_t chunk_start = 0;
                    for (int prev_chunk = 0; prev_chunk < chunk_idx; prev_chunk++) {
                        chunk_start += token_column->chunk(prev_chunk)->length();
                    }

                    if (row >= chunk_start && row < chunk_start + list_array->length()) {
                        int64_t local_row = row - chunk_start;

                        if (!list_array->IsNull(local_row)) {
                            auto slice = list_array->value_slice(local_row);
                            if (llama_dataset_arrow_array_to_tokens(slice, sequence_tokens)) {
                                sequence_processed = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        // If no tokenized data found, try to tokenize text data
        if (!sequence_processed && format_data->schema_info.primary_text_column_index >= 0 && format_data->tokenizer) {
            auto text_column = table->column(format_data->schema_info.primary_text_column_index);

            // Get the text data for this row
            for (int chunk_idx = 0; chunk_idx < text_column->num_chunks(); chunk_idx++) {
                auto chunk = text_column->chunk(chunk_idx);

                if (chunk->type_id() == arrow::Type::STRING || chunk->type_id() == arrow::Type::LARGE_STRING) {
                    auto string_array = std::static_pointer_cast<arrow::StringArray>(chunk);

                    // Find the row within this chunk
                    int64_t chunk_start = 0;
                    for (int prev_chunk = 0; prev_chunk < chunk_idx; prev_chunk++) {
                        chunk_start += text_column->chunk(prev_chunk)->length();
                    }

                    if (row >= chunk_start && row < chunk_start + string_array->length()) {
                        int64_t local_row = row - chunk_start;

                        if (!string_array->IsNull(local_row)) {
                            std::string text = string_array->GetString(local_row);
                            if (!text.empty()) {
                                sequence_tokens = format_data->tokenizer->tokenize_text(text);
                                if (!sequence_tokens.empty()) {
                                    sequence_processed = true;
                                    break;
                                } else {
                                    handle_tokenization_error(row, "text", text, "Mixed content tokenization failed");
                                }
                            }
                        }
                    }
                }
            }
        }

        // Add the processed sequence if we got one
        if (sequence_processed && !sequence_tokens.empty()) {
            all_sequences.push_back(std::move(sequence_tokens));
            max_length = std::max(max_length, static_cast<int32_t>(all_sequences.back().size()));
        } else if (!sequence_processed) {
            LLAMA_LOG_WARN("No valid data found for row %ld in mixed content processing\n", row);
        }
    }

    return true;
}

#endif // LLAMA_PARQUET
